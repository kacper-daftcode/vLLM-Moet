"""Does the pool want to run a layer's 2 misses IN PARALLEL on two half-pools instead of one after another on the full
pool? The decode's wait tail is layers with 2-3 misses (note: the per-layer wait median is 8-9 us and only ~10 layers
per step miss, but those carry 150 us each), and run_job() walks the job's experts one at a time on all 6 threads.

The kernel's cost at m = 1 is dominated by streaming the record's bytes (45 GB/s of 4.7 MB), so a half-pool gets ~half
the bandwidth: 2 experts on 2 x 3 threads ~ 2 x (2 x t_3threads) against 2 x t_6threads. This measures it on REAL
records with the decode's job pattern. No GPU, so only run this when no TP4 boot of ours is up.

    python bench_expert_parallel.py 10-15 insitu        # 6 threads, 1 and 2 experts per job, the current path
    python bench_expert_parallel.py 10-12,13-15 insitu  # the same records split over two half-pools
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CODE = r'''
import sys, time, os, numpy as np, torch, ctypes
sys.path.insert(0, %r)
import dsv41_b12x as B
from test_cpu_moe_b12x import realistic_hidden
torch.set_num_threads(1)
groups = [g for g in os.environ["GROUPS"].split(";")]
npool = len(groups)
SECS = float(os.environ.get("SECS", "6"))
GAP = float(os.environ.get("GAP_US", "250")); PER = int(os.environ.get("PER", "2"))
TSC = float(os.environ.get("DSV41_TSC_GHZ", "3.9937"))
B.load_lib()
g = torch.Generator().manual_seed(3)
spec = os.environ.get("RECORDS", "")
N = 48
if spec:
    store, layer, rank = spec.split(":")
    path = os.path.join(store, "rank%%d" %% int(rank), "layer%%02d.bin" %% int(layer))
    with open(path, "rb") as f:
        recs = [np.frombuffer(f.read(B.RECORD_BYTES), dtype=np.uint8) for _ in range(N)]
else:
    recs = [B.random_record(g, 116 + 2 * (i %% 4), 124 + 2 * (i %% 4)) for i in range(N)]
x = realistic_hidden(6, g, "realistic"); aq, as_ = B.rows_quant(x)
arena = B.RecordArena(N)
for i in range(N): arena.slot(i)[:] = recs[i]
ptrs = [arena.ptr(i) for i in range(N)]
out = np.zeros((N, B.N2), dtype=np.uint16)
pools = []
for grp in groups:
    cpus = [int(x) for x in grp.split(",")]
    pools.append((cpus, B.Pool(cpus)))
def job(idx):
    """idx: a list of experts for THIS job; split over the pools, each pool getting whole experts"""
    if npool == 1:
        pools[0][1].experts_forward([ptrs[i] for i in idx], [1]*len(idx), list(range(len(idx))), [0.5]*len(idx),
                                    list(range(len(idx))), aq, as_, out)
    else:
        k = len(idx)
        for p, (cpus, pool) in enumerate(pools):
            mine = [e for j, e in enumerate(idx) if j %% npool == p]
            if not mine: continue
            pool.experts_forward([ptrs[i] for i in mine], [1]*len(mine), list(range(len(mine))), [0.5]*len(mine),
                                 list(range(len(mine))), aq, as_, out)
def busy(us):
    t = time.perf_counter()
    while (time.perf_counter() - t) * 1e6 < us: pass
os.sched_setaffinity(0, set(sum([c for c, _ in pools], [])) | {int(os.environ["MAIN_CPU"])})
job(list(range(min(4, N))))
print("READY", flush=True)
flag = os.environ["START_FLAG"]
while not os.path.exists(flag): time.sleep(0.001)
t0 = time.perf_counter(); n = 0; lat = 0.0; njobs = 0
while time.perf_counter() - t0 < SECS:
    for j in range(30):
        i0 = (njobs * PER) %% N
        idx = [(i0 + k) %% N for k in range(PER)]
        t1 = time.perf_counter(); job(idx); lat += time.perf_counter() - t1; n += PER; njobs += 1
        busy(GAP)
    busy(4000)
dt = time.perf_counter() - t0
tot_cyc = 0; per_pool = []
for _, p in pools:
    s = p.stats(reset=True)
    per_pool.append((s["cyc_fc1"] + s["cyc_fc2"] + s["cyc_prepare"] + s["cyc_act"]) / TSC / 1e3 / n)
print(f"GROUPS {os.environ['GROUPS']} PER {PER}: {lat / n * 1e6:.1f} us per expert in the job; "
      f"in-kernel per expert {sum(per_pool):.1f} (pools {[round(x, 1) for x in per_pool]}); {n} experts / {njobs} jobs in {dt:.1f} s")
for cpus, pool in pools: pool.close()
arena.close()
''' % HERE


def main() -> int:
    groups = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "insitu"
    secs = sys.argv[3] if len(sys.argv) > 3 else "6"
    flag = f"/tmp/bench_ep_start_{os.getpid()}"
    if os.path.exists(flag):
        os.remove(flag)
    first = groups.split(";")[0].split(",")
    env = dict(os.environ, GROUPS=groups, MODE=mode, SECS=secs, START_FLAG=flag, MAIN_CPU=str((int(first[0]) // 8) * 8))
    print(f"== {groups} (PER={env.get('PER', '2')}, {mode}, {secs}s)", flush=True)
    p = subprocess.Popen([sys.executable, "-c", CODE], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = p.stdout.readline()
    while line and "READY" not in line:
        print(line.rstrip())
        line = p.stdout.readline()
    open(flag, "w").close()
    out, _ = p.communicate()
    print(out.strip())
    os.remove(flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
