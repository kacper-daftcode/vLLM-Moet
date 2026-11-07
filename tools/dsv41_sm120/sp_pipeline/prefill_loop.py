#!/usr/bin/env python3
"""Fresh 163K prefills back to back (4600 lines with random ids, max_tokens 1), the load under which the decode streams
of mixed_steps.sh run. Stops after N prefills or when the stop file appears.
usage: prefill_loop.py BASE MODEL N STOPFILE [--lines 4600] [--seed-first SEED --dump-dir DIR]
  --seed-first: the first prefill uses a deterministic prompt (dump_prefill.py's recipe, seeded ids) and, with
  --dump-dir, DIR/ENABLE is present during it alone, so the server's per-layer dumps describe that one prefill."""
import json
import os
import random
import sys
import time
import urllib.request
import uuid

BASE, MODEL, N, STOP = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
lines = int(sys.argv[sys.argv.index("--lines") + 1]) if "--lines" in sys.argv else 4600
seed_first = int(sys.argv[sys.argv.index("--seed-first") + 1]) if "--seed-first" in sys.argv else None
dump_dir = sys.argv[sys.argv.index("--dump-dir") + 1] if "--dump-dir" in sys.argv else ""
for i in range(N):
    if os.path.exists(STOP):
        print(f"[{time.strftime('%H:%M:%S')}] stop file seen after {i} prefills", flush=True)
        break
    if i == 0 and seed_first is not None:
        rng = random.Random(seed_first)
        text = "".join(f"[{rng.getrandbits(32):08x}] Line {j}: the scheduler assigns blocks to requests and reclaims them when sequences finish; prefix caching reuses identical leading blocks across requests.\n" for j in range(lines))
    else:
        text = "".join(f"[{uuid.uuid4().hex[:8]}] Line {j}: the scheduler assigns blocks to requests and reclaims them when sequences finish; prefix caching reuses identical leading blocks across requests.\n" for j in range(lines))
    flag = os.path.join(dump_dir, "ENABLE") if (dump_dir and i == 0 and seed_first is not None) else ""
    if flag:
        os.makedirs(dump_dir, exist_ok=True)
        open(flag, "w").close()
    body = {"model": MODEL, "messages": [{"role": "user", "content": text + "\nReply with one word."}], "max_tokens": 1,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    finally:
        if flag and os.path.exists(flag):
            time.sleep(2)
            os.remove(flag)
    dt = time.perf_counter() - t0
    pin = d["usage"]["prompt_tokens"]
    print(f"[{time.strftime('%H:%M:%S')}] fresh prefill {pin} tok: {dt*1000:.0f} ms = {pin/dt:.0f} tok/s{' (seeded, dumped)' if flag else ''}", flush=True)
