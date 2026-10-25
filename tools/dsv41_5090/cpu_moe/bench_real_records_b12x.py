"""Kernel v2 (cpu_moe_b12x) on REAL expert records from the store files against the synthetic uniform-nibble records
the other benches use. Real weights cost more in FC2 (the data-dependent alignment prologue: the rare path is twice
as frequent, the branches mispredict) - the synthetic bench under-reports the in-situ cost by ~1.3x (10-24).

    python bench_real_records_b12x.py [--store /mnt/nvme/packs/dsv41-hybrid] [--layer 20] [--rank 0] [--experts 24]
                                      [--cpus 10-15] [--per-job 24,1] [--m 1,2,6] [--secs 3]

Per variant: us per expert (wall, over the jobs), worker 0's TSC time inside the kernel per expert, the FC1 / FC2
cycle split, the FC2 general-path fraction (row-blocks through the rare recomputation) and the q0 fraction (4-row
steps that truncate nothing - what an int8 VNNI dot could take). Records live in a THP-advised arena (the shape of
the pinned store); 4 KB pages cost 2-7 % more (10-24).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_b12x as B  # noqa: E402
from test_cpu_moe_b12x import realistic_hidden  # noqa: E402

TSC_GHZ = float(os.environ.get("DSV41_TSC_GHZ", "3.9937"))   # rtxdev2: 3993.689 MHz


def real_records(store: str, layer: int, rank: int, experts: list[int]) -> list[np.ndarray]:
    path = os.path.join(store, f"rank{rank}", f"layer{layer:02d}.bin")
    out = []
    with open(path, "rb") as f:
        for e in experts:
            f.seek(e * B.RECORD_BYTES)
            rec = np.frombuffer(f.read(B.RECORD_BYTES), dtype=np.uint8)
            assert rec.size == B.RECORD_BYTES, (path, e)
            out.append(rec)
    return out


def run(pool: B.Pool, arena: B.RecordArena, n: int, m: int, per_job: int, aq, as_, secs: float) -> dict:
    out = np.zeros((n * m, B.N2), dtype=np.uint16)
    ptrs = [arena.ptr(i) for i in range(n)]

    def job(idx):
        pool.experts_forward([ptrs[i] for i in idx], [m] * len(idx), [t % aq.shape[0] for _ in idx for t in range(m)],
                             [0.5] * (len(idx) * m), [i * m + t for i in idx for t in range(m)], aq, as_, out)

    job(list(range(n)))   # warm (pages touched, the pool awake)
    pool.stats(reset=True)
    t0 = time.perf_counter()
    done = 0
    while time.perf_counter() - t0 < secs:
        for j0 in range(0, n, per_job):
            idx = list(range(j0, min(n, j0 + per_job)))
            job(idx)
            done += len(idx)
    dt = time.perf_counter() - t0
    st = pool.stats(reset=True)
    cyc = st["cyc_prepare"] + st["cyc_fc1"] + st["cyc_act"] + st["cyc_fc2"]
    return dict(us=dt / done * 1e6, in_kernel=cyc / TSC_GHZ / 1e3 / done, fc1=100 * st["cyc_fc1"] / max(1, cyc),
                fc2=100 * st["cyc_fc2"] / max(1, cyc), general=100 * st["fc2_general"] / max(1, st["fc2_blocks"]),
                q0=100 * st.get("fc2_q0", 0) / max(1, st["fc2_blocks"]), experts=done)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/mnt/nvme/packs/dsv41-hybrid")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--first-expert", type=int, default=0)
    ap.add_argument("--cpus", default="10-15")
    ap.add_argument("--per-job", default="24,1")
    ap.add_argument("--m", default="1,2,6")
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--no-synthetic", action="store_true")
    args = ap.parse_args()
    a, b = args.cpus.split("-")
    cpus = list(range(int(a), int(b) + 1))
    os.sched_setaffinity(0, {(cpus[0] // 8) * 8})   # the main thread on the CCD's first core, as in situ
    B.load_lib()
    g = torch.Generator().manual_seed(3)
    x = realistic_hidden(6, g, "realistic")
    inputs = os.path.join(B.TABLES_DIR, "inputs.npz")
    if os.path.exists(inputs):
        Z = np.load(inputs)
        x = torch.from_numpy(Z["x_bf16"].astype(np.int16)).view(torch.bfloat16)[:6]
        src = "recorded hidden states"
    else:
        src = "synthetic realistic hidden states"
    aq, as_ = B.rows_quant(x)
    n = args.experts
    experts = list(range(args.first_expert, args.first_expert + n))
    variants = []
    if not args.no_synthetic:
        variants.append(("synthetic (uniform nibbles, scales 116-126)", [B.random_record(g, 116 + 2 * (i % 4), 124 + 2 * (i % 4)) for i in range(n)]))
    t0 = time.perf_counter()
    variants.append((f"real (store layer {args.layer} rank {args.rank} experts {experts[0]}-{experts[-1]})",
                     real_records(args.store, args.layer, args.rank, experts)))
    print(f"{n} records per variant ({n * B.RECORD_BYTES / 2**20:.0f} MiB arena), {src} ({x.shape[0]} tokens), CPUs {args.cpus} "
          f"({len(cpus)} threads); store read {time.perf_counter() - t0:.1f} s")
    pool = B.Pool(cpus)
    arena = B.RecordArena(n)
    for name, recs in variants:
        for i in range(n):
            arena.slot(i)[:] = recs[i]
        print(f"-- {name}")
        for per_job in [int(v) for v in args.per_job.split(",")]:
            for m in [int(v) for v in args.m.split(",")]:
                r = run(pool, arena, n, m, per_job, aq, as_, args.secs)
                print(f"  m {m} {per_job:2d} experts/job: {r['us']:6.1f} us per expert (worker-0 in-kernel {r['in_kernel']:6.1f}); "
                      f"fc1 {r['fc1']:.0f} % fc2 {r['fc2']:.0f} %; fc2 general path {r['general']:.1f} % of row-blocks, "
                      f"q0 steps {r['q0']:.1f} %", flush=True)
    arena.close()
    pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
