"""Is kernel v2 memory-bound or compute-bound per core, and what do the decode's idle gaps cost which phase? The same
pool, the same real records (layer 20, rank 0), jobs of one expert: (A) cycling over 48 distinct records (226 MB:
every expert from DRAM) against (B) the same single record every job (4.7 MB: L3-resident after the first job);
interleaved A/B per rep, SECS each, the per-phase in-kernel split (prepare / FC1 / act / FC2) from worker 0's counters.
GAP_US > 0 puts a busy gap on the main thread before every job (the decode's pattern). The box must be quiet on the
pool's CCD (mpstat the cores first); no ds41-hyb boot of ours up (rule: no CPU benches under a TP4 boot).

    CPUS=10,11,12,13,14,15 MAIN_CPU=8 SECS=3 [GAP_US=250] [REPS=3] [MOE_B12X_PREFETCH=1 MOE_B12X_PREFETCH_HINT=1]
      [MOE_B12X_POOL_SPIN_HOT=2 MOE_B12X_POOL_SPIN_TOUCH_STRIDE=64] python bench_l3_vs_dram_b12x.py

10-29 (CCD 1, 6 threads): A 119 us in-kernel (FC1 58, FC2 58), B 69 (36 / 30) - 40 % memory stalls, not bandwidth
(ccd_bw.cpp: one CCD streams 55-57 GB/s = 82 us per record). With GAP_US=250 only FC1 slows (54 -> 75), B does not:
the after-a-gap penalty is a per-core memory-path state (note §20). MOE_B12X_PREFETCH=1: A 110 (-11 %). Prefetch +
POOL_SPIN_HOT=2 with the sequential touch stream: the gap case 151 -> 120.5 us per expert (-20 %).
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_b12x as B
from test_cpu_moe_b12x import realistic_hidden

torch.set_num_threads(1)
cpus = [int(x) for x in os.environ.get("CPUS", "10,11,12,13,14,15").split(",")]
SECS = float(os.environ.get("SECS", "3"))
TSC = float(os.environ.get("DSV41_TSC_GHZ", "3.9937"))
N = 48
B.load_lib()
g = torch.Generator().manual_seed(3)
store, layer, rank = "/mnt/nvme/packs/dsv41-hybrid", 20, 0
with open(os.path.join(store, f"rank{rank}", f"layer{layer:02d}.bin"), "rb") as f:
    recs = [np.frombuffer(f.read(B.RECORD_BYTES), dtype=np.uint8) for _ in range(N)]
x = realistic_hidden(6, g, "realistic"); aq, as_ = B.rows_quant(x)
arena = B.RecordArena(N)
for i in range(N): arena.slot(i)[:] = recs[i]
os.sched_setaffinity(0, {int(os.environ.get("MAIN_CPU", "8"))})
pool = B.Pool(cpus)
ptrs = [arena.ptr(i) for i in range(N)]
out = np.zeros((N, B.N2), dtype=np.uint16)

def job(i):
    pool.experts_forward([ptrs[i]], [1], [0], [0.5], [i], aq, as_, out)

GAP = float(os.environ.get("GAP_US", "0"))
def busy(us):
    t = time.perf_counter()
    while (time.perf_counter() - t) * 1e6 < us: pass
def run(label, seq):
    job(seq[0]); pool.stats(reset=True)
    t0 = time.perf_counter(); n = 0; lat = 0.0; k = 0
    while time.perf_counter() - t0 < SECS:
        i = seq[k % len(seq)]; k += 1
        if GAP > 0: busy(GAP)
        t1 = time.perf_counter(); job(i); lat += time.perf_counter() - t1; n += 1
    st = pool.stats(reset=True)
    cyc = st["cyc_prepare"] + st["cyc_fc1"] + st["cyc_act"] + st["cyc_fc2"]
    fc1 = st["cyc_fc1"] / TSC / 1e3 / n; fc2 = st["cyc_fc2"] / TSC / 1e3 / n
    pre = st["cyc_prepare"] / TSC / 1e3 / n; act = st["cyc_act"] / TSC / 1e3 / n
    print(f"gap {GAP:4.0f} {label:>14}: {lat / n * 1e6:6.1f} us per expert (worker-0 in-kernel {cyc / TSC / 1e3 / n:6.1f}: prep {pre:4.1f} fc1 {fc1:5.1f} act {act:4.1f} fc2 {fc2:5.1f}); n {n}", flush=True)

for rep in range(int(os.environ.get("REPS", "3"))):
    run("A dram 48 rec", list(range(N)))
    run("B l3 1 rec", [7])
pool.close(); arena.close()
