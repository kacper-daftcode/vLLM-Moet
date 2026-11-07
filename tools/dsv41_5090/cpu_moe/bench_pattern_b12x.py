"""Kernel v2 under the decode's job pattern, one process per CCD (the four ranks' pools), with the pool cores'
effective frequency from perf (cycles per second over the run).

    python bench_pattern_b12x.py "10,11,12,13,14,15" insitu            # one CCD
    python bench_pattern_b12x.py "2,3,4,5,6,7;10,11,12,13,14,15;18,19,20,21,22,23;26,27,28,29,30,31" insitu [SECS]
    MODE  b2b     1 expert per job, back to back
          insitu  PER (2) experts per job, a busy host gap of GAP_US (250) between jobs, BIG_US (4000) every 30 jobs
                  = the decode's pattern (served layers ~250 us apart, the step boundary + host hook in between)
    env   MOE_B12X_POOL_SPIN_US (the pool's idle policy; -1 = never sleep), PER, GAP_US, BIG_US, RECORDS (real: layer
          records from the store, e.g. RECORDS=/mnt/nvme/packs/dsv41-hybrid:20:0 = store:layer:rank; default synthetic)

10-25 (CCD 1 alone / 4 CCDs at once, 6 threads each, synthetic records): b2b 127 / 137-139 us per expert; the
decode pattern 138 / 140 (spin -1: 133); 1 per job 150. The job pattern and the cross-CCD DRAM contention cost
+8-10 % each - the in-situ 207-300 us of the 10-22..24 boots were the real weights' FC2 cost (bench_real_records).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CODE = r'''
import sys, time, os, numpy as np, torch
sys.path.insert(0, %r)
import dsv41_b12x as B
from test_cpu_moe_b12x import realistic_hidden
torch.set_num_threads(1)   # the main thread is pinned to one core below: no OpenMP pool fighting over it
cpus = [int(x) for x in os.environ["CPUS"].split(",")]
N = 48; MODE = os.environ["MODE"]; SECS = float(os.environ.get("SECS", "6"))
PER = int(os.environ.get("PER", "2")); GAP = float(os.environ.get("GAP_US", "250")); BIG = float(os.environ.get("BIG_US", "4000"))
TSC = float(os.environ.get("DSV41_TSC_GHZ", "3.9937"))
B.load_lib()
g = torch.Generator().manual_seed(3)
spec = os.environ.get("RECORDS", "")
if spec:
    store, layer, rank = spec.split(":")
    path = os.path.join(store, f"rank{int(rank)}", f"layer{int(layer):02d}.bin")
    with open(path, "rb") as f:
        recs = [np.frombuffer(f.read(B.RECORD_BYTES), dtype=np.uint8) for _ in range(N)]
else:
    recs = [B.random_record(g, 116 + 2 * (i %% 4), 124 + 2 * (i %% 4)) for i in range(N)]
x = realistic_hidden(6, g, "realistic"); aq, as_ = B.rows_quant(x)
arena = B.RecordArena(N)
for i in range(N): arena.slot(i)[:] = recs[i]
os.sched_setaffinity(0, {int(os.environ["MAIN_CPU"])})   # the rank's main-thread core, as in situ
pool = B.Pool(cpus)
ptrs = [arena.ptr(i) for i in range(N)]
out = np.zeros((N, B.N2), dtype=np.uint16)
def job(idx):
    pool.experts_forward([ptrs[i] for i in idx], [1] * len(idx), [0] * len(idx), [0.5] * len(idx), list(idx), aq, as_, out)
def busy(us):
    t = time.perf_counter()
    while (time.perf_counter() - t) * 1e6 < us: pass
job(list(range(N)))
print("READY", flush=True)
flag = os.environ["START_FLAG"]
while not os.path.exists(flag): time.sleep(0.001)
pool.stats(reset=True)
t0 = time.perf_counter(); n = 0; njobs = 0; lat = 0.0
while time.perf_counter() - t0 < SECS:
    if MODE == "b2b":
        for i in range(N):
            t1 = time.perf_counter(); job([i]); lat += time.perf_counter() - t1; n += 1; njobs += 1
    else:
        for j in range(30):
            i0 = (njobs * PER) %% N
            idx = [(i0 + k) %% N for k in range(PER)]
            t1 = time.perf_counter(); job(idx); lat += time.perf_counter() - t1
            n += PER; njobs += 1
            busy(GAP)
        busy(BIG)
dt = time.perf_counter() - t0
st = pool.stats(reset=True)
cyc = st["cyc_prepare"] + st["cyc_fc1"] + st["cyc_act"] + st["cyc_fc2"]
print(f"CPUS {os.environ['CPUS']} {MODE}: {lat / n * 1e6:.1f} us per expert in the job (worker-0 in-kernel {cyc / TSC / 1e3 / n:.1f}); "
      f"{n} experts / {njobs} jobs in {dt:.1f} s; duty {100 * lat / dt:.0f} %%", flush=True)
pool.close(); arena.close()
''' % HERE


def main() -> int:
    sets = sys.argv[1].split(";")
    mode = sys.argv[2] if len(sys.argv) > 2 else "insitu"
    secs = sys.argv[3] if len(sys.argv) > 3 else "6"
    flag = f"/tmp/bench_pattern_start_{os.getpid()}"
    if os.path.exists(flag):
        os.remove(flag)
    procs, all_cpus = [], []
    for s in sets:
        cpus = [int(x) for x in s.split(",")]
        all_cpus += cpus
        env = dict(os.environ, CPUS=s, MODE=mode, SECS=secs, START_FLAG=flag, MAIN_CPU=str((cpus[0] // 8) * 8))
        procs.append(subprocess.Popen([sys.executable, "-c", CODE], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    for p in procs:   # every process has set up and warmed; then they start together
        line = p.stdout.readline()
        while line and "READY" not in line:
            print(line.rstrip())
            line = p.stdout.readline()
    open(flag, "w").close()
    time.sleep(1.0)
    dur = max(1.0, float(secs) - 2.5)
    ps = subprocess.run(["perf", "stat", "-A", "-C", ",".join(map(str, all_cpus)), "-e", "cycles", "-x", ",", "--", "sleep", str(dur)],
                        capture_output=True, text=True)
    for p in procs:
        out, _ = p.communicate()
        print(out.strip())
    os.remove(flag)
    freq = {}
    for l in ps.stderr.strip().splitlines():
        f = l.split(",")
        if len(f) >= 4 and f[0].startswith("CPU") and f[3] == "cycles" and f[1].isdigit():
            freq[int(f[0][3:])] = int(f[1]) / dur / 1e9
    if freq:
        fs = sorted(freq.values())
        print(f"  pool cores during the run: {fs[0]:.2f}-{fs[-1]:.2f} GHz effective (median {fs[len(fs) // 2]:.2f}; "
              f"cycles / {dur:.1f} s, idle included - at 100 % duty the running clock)")
    else:
        print(f"  perf stat: {ps.stderr.strip()[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
