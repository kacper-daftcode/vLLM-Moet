"""The CPU expert tier with VIRTUAL RANKS (TP2 / TP1 on one GPU, 2026-11-04) end to end on one GPU, without vLLM:
V tiers in one process (one record set, pool and mailbox set per 576-row slice), the expanded routing [T, K * V],
one publish_v launch -> (GPU stand-in for the resident slice-experts) -> one wait_inject_v launch, eagerly and from
one CUDA graph per step, against the b12x-exact reference per (pair, slice); fail-closed.

In the serving image, one sm_120 GPU under its lock, e.g.
    flock -w 3600 /tmp/exl3diff-gpu1.lock docker run --rm --gpus '"device=1"' --name moet-tier-test \
      -v <cache dir>/dot-cache:/root/.cache -v <checkout>:/repo -v <parity dir>:/parity \
      --entrypoint python3 <image> /repo/tools/dsv41_5090/cpu_tier/test_cpu_tier_vranks.py \
      --records /parity/b12x_inputs.npz,/parity/b12x_inputs_l0.npz,/parity/b12x_inputs_l1.npz,/parity/b12x_inputs_l39.npz \
      --tables /parity --layers 4 --steps 200 --cpus "8-13;16-21;24-29;2-7" --dispatcher-cpus 40,48,56,33

What it checks (the V = 1 test is test_cpu_tier.py; nothing of it is changed by the virtual ranks):
  1. eager steps: every missing pair's V route rows == the scalar reference of slice v's record (bit for bit); the
     resident pairs' rows (the stand-in's) untouched; the use log carries the (token, expert) pairs once
  2. a captured step (step counter + L x (publish_v, stand-in, wait_inject_v)) replayed with fresh routings: the same
     identity on sampled steps; GPU time per step with / without the tier's ops
  3. fail-closed: one virtual rank's dispatcher stopped -> one replay arms the error word within the timeout; the
     next replay short-circuits; restart -> correct again
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "cpu_moe"))

from test_cpu_tier import StandIn, dequant_fp4, parse_cpus  # noqa: E402,F401


def ensure_libs(args):
    if os.environ.get("CPU_TIER_LIB"):
        return
    out = os.path.join(args.build_dir, "libcpu_tier.so")
    if not os.path.exists(out) or args.rebuild:
        env = dict(os.environ, ARCH=args.arch, OUT=out)
        subprocess.run(["bash", os.path.join(_HERE, "build.sh")], check=True, env=env)
    os.environ["CPU_TIER_LIB"] = out
    os.environ["DSV41_B12X_LIB"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="comma-separated b12x_inputs*.npz, one per virtual rank (its slice's records)")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--cpus", default="8-13;16-21", help="';'-separated pool CPU ranges, one per virtual rank")
    ap.add_argument("--dispatcher-cpus", default="40,48")
    ap.add_argument("--timeout-ms", type=float, default=500.0)
    ap.add_argument("--build-dir", default="/tmp")
    ap.add_argument("--arch", default="znver4")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ensure_libs(args)
    os.environ["DSV41_B12X_TABLES"] = args.tables
    import dsv41_b12x as B  # noqa: E402
    from cpu_tier import TierGroup  # noqa: E402

    dev = torch.device("cuda", 0)
    rec_files = args.records.split(",")
    pools = [parse_cpus(p) for p in args.cpus.split(";")]
    disps = [int(x) for x in args.dispatcher_cpus.split(",")]
    V = len(rec_files)
    assert len(pools) == V and len(disps) == V, "one record file, pool and dispatcher per virtual rank"
    recs = [np.load(f)["records"] for f in rec_files]
    E = min(r.shape[0] for r in recs)
    recs = [r[:E] for r in recs]
    T, K, L = args.tokens, args.topk, args.layers
    KV = K * V
    assert T <= 8
    stores = [torch.from_numpy(r).pin_memory() for r in recs]
    # the stand-in's slice-experts: row e * V + v = (expert e, slice v)
    split = {k: torch.stack([torch.from_numpy(np.ascontiguousarray(B.split_record(recs[v][e])[k])) for e in range(E) for v in range(V)])
             for k in ("w1n", "w1s", "w3n", "w3s", "w2n", "w2s")}
    stand_in = StandIn(split, dev)
    print(f"virtual ranks V={V}: {E} experts x {V} slice records ({V * E * B.RECORD_BYTES / 2**20:.0f} MiB pinned); layers {L}, "
          f"tokens {T}, top-{K} -> {KV} pairs per token; pools {pools}, dispatchers {disps}", flush=True)

    tier = TierGroup(L, E, pools, disps, dev, args.tables, timeout_ms=args.timeout_ms)
    for layer in range(L):
        for e in range(E):
            for v in range(V):
                tier.set_record(layer, e, stores[v].data_ptr() + e * B.RECORD_BYTES, v)
    # resident subsets per layer: the experts with (e + l) % 3 != 0; slot s holds slice-experts s * V + v
    slot_tables = []
    for layer in range(L):
        slot = torch.full((E,), -1, dtype=torch.int32)
        k = 0
        for e in range(E):
            if (e + layer) % 3 != 0:
                slot[e] = k
                k += 1
        slot_tables.append(slot.to(dev))
    # the stand-in indexes slice-experts by e * V + v; the resident pool would hold slot * V + v - the stand-in gets the
    # EXPERT ids expanded (ids_x) and the remapped ids only for the keep mask, so both layouts agree on what is computed
    tier.start(first_step=1)

    x_bf16 = torch.zeros(T, B.HID, dtype=torch.bfloat16, device=dev)
    a_values = torch.zeros(16, B.HID, dtype=torch.uint8, device=dev)
    a_scales = torch.zeros(16, B.B1, dtype=torch.uint8, device=dev)
    ids = torch.zeros(T, K, dtype=torch.int32, device=dev)
    wts = torch.zeros(T, K, dtype=torch.float32, device=dev)
    ids_x = torch.zeros(T, KV, dtype=torch.int32, device=dev)       # the expanded routing (expert ids, V times)
    wts_x = torch.zeros(T, KV, dtype=torch.float32, device=dev)
    vidx = (torch.arange(KV, dtype=torch.int32, device=dev) % V).unsqueeze(0)
    stand_ids = torch.zeros(T, KV, dtype=torch.int32, device=dev)   # e * V + v for the stand-in
    remapped = [torch.zeros(T, KV, dtype=torch.int32, device=dev) for _ in range(L)]
    route = [torch.zeros(T * KV, B.HID, dtype=torch.bfloat16, device=dev) for _ in range(L)]
    g = torch.Generator().manual_seed(7)

    def new_inputs():
        x = torch.randn(T, B.HID, generator=g)
        heavy = torch.randperm(B.HID, generator=g)[:48]
        x[:, heavy] *= 12.0
        xb = x.to(torch.bfloat16)
        q, s = B.rows_quant(xb, zero_byte=0)
        i = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(torch.int32)
        w = torch.rand(T, K, generator=g) * 0.9 + 0.1
        x_bf16.copy_(xb)
        a_values[:T].copy_(torch.from_numpy(q))
        a_scales[:T].copy_(torch.from_numpy(s))
        ids.copy_(i)
        wts.copy_(w)
        return q, s, i.numpy(), w.numpy()

    def step(with_tier=True):
        if with_tier:
            tier.step_begin()
        # the expansion as the runtime does it (one copy kernel each into static buffers)
        ids_x.view(T, K, V).copy_(ids.unsqueeze(-1).expand(T, K, V))
        wts_x.view(T, K, V).copy_(wts.unsqueeze(-1).expand(T, K, V))
        stand_ids.copy_(ids_x * V + vidx)
        xf = x_bf16.float()
        for layer in range(L):
            if with_tier:
                tier.publish(layer, ids_x, wts_x, slot_tables[layer], a_values, a_scales, T, remapped[layer])
            else:
                slots = slot_tables[layer][ids_x.long()]
                remapped[layer].copy_(torch.where(slots >= 0, slots * V + vidx, -1))
            stand_in(xf, stand_ids, remapped[layer], wts_x, route[layer])
            if with_tier:
                tier.wait_inject(layer, route[layer])

    def verify(q, s, i_np, w_np, label):
        torch.cuda.synchronize()
        bad = 0
        n_missing = 0
        for layer in range(L):
            slot = slot_tables[layer].cpu().numpy()
            rows = route[layer].view(torch.int16).cpu().numpy().view(np.uint16)
            rm = remapped[layer].cpu().numpy().reshape(-1)
            keep_rows = route[layer].clone()
            stand_in(x_bf16.float(), stand_ids, remapped[layer], wts_x, keep_rows)
            keep = keep_rows.view(torch.int16).cpu().numpy().view(np.uint16)
            for t in range(T):
                for k in range(K):
                    e = int(i_np[t, k])
                    for v in range(V):
                        p = (t * K + k) * V + v
                        if slot[e] >= 0:
                            bad += int(rm[p] != slot[e] * V + v)
                            bad += int(not np.array_equal(rows[p], keep[p]))
                        else:
                            bad += int(rm[p] != -1)
                            n_missing += 1
                            ref = B.reference_pair(recs[v][e], q[t], s[t], float(w_np[t, k]))
                            d = int((rows[p] != ref["out"]).sum())
                            bad += d > 0
                            if d and bad <= 3:
                                print(f"    {label}: layer {layer} pair {p} (expert {e}, slice {v}) differs in {d} columns", flush=True)
        return bad, n_missing

    # ---- 1. eager -----------------------------------------------------------------------------------------------
    total_bad = total_missing = 0
    for it in range(3):
        q, s, i_np, w_np = new_inputs()
        step()
        bad, nm = verify(q, s, i_np, w_np, f"eager {it}")
        total_bad += bad
        total_missing += nm
    err = tier.error()
    uses = tier.uses()
    # one (token, expert) id per pair, not per slice: T * K ids per (step, layer), all valid expert ids, the last
    # entry = the last step's routing
    use_ok = all(n_ids.shape[0] == T * K and n_ids.min() >= 0 and n_ids.max() < E for _, _, n_ids in uses) \
        and np.array_equal(uses[-1][2].astype(np.int64), i_np.reshape(-1).astype(np.int64))
    print(f"1. eager: {total_missing} missing (pair, slice) rows over 3 steps x {L} layers, rows / remaps differing from the "
          f"reference: {total_bad}; error word {err}; use log entries {len(uses)} ({T * K} ids each: {use_ok})", flush=True)
    ok1 = total_bad == 0 and err == 0 and len(uses) == 3 * L and use_ok

    # ---- 2. captured ----------------------------------------------------------------------------------------------
    sgraph = torch.cuda.Stream()
    sgraph.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sgraph):
        new_inputs()
        step()
    torch.cuda.current_stream().wait_stream(sgraph)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    new_inputs()
    with torch.cuda.graph(graph):
        step()
    graph_plain = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_plain):
        step(with_tier=False)
    torch.cuda.synchronize()
    ok2 = True
    times, times_plain = [], []
    checked = bad2 = missing2 = 0
    for it in range(args.steps):
        q, s, i_np, w_np = new_inputs()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        if (it + 1) % args.check_every == 0 or it < 2:
            bad, nm = verify(q, s, i_np, w_np, f"replay {it}")
            checked += 1
            bad2 += bad
            missing2 += nm
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
        if tier.error():
            print(f"    error word {tier.error()} after replay {it}", flush=True)
            ok2 = False
            break
    torch.cuda.synchronize()
    for it in range(50):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph_plain.replay()
        e1.record()
        e1.synchronize()
        times_plain.append(e0.elapsed_time(e1))
    tot = tier.totals()
    per_v = []
    for v in range(V):
        kp = tier.kpis_v(v)
        per_v.append(f"v{v}: busy median {np.median((kp[:, 6] - kp[:, 5]) / 1e3):.0f} us, pairs median {np.median(kp[:, 2]):.0f}")
    print(f"2. captured step replayed {len(times)} x: GPU time per step median {np.median(times):.3f} ms (plain stand-in "
          f"{np.median(times_plain):.3f} ms -> tier overhead {np.median(times) - np.median(times_plain):.3f} ms per step for {L} "
          f"layers x {V} virtual ranks incl. the CPU time); checked {checked} steps: {missing2} missing rows, {bad2} wrong; "
          f"totals steps {tot['steps']} pairs {tot['pairs']} errors {tot['errors']} busy_v {[round(b / 1e6) for b in tot['busy_ns_v']]} ms; "
          f"{'; '.join(per_v)}", flush=True)
    ok2 = ok2 and bad2 == 0 and tier.error() == 0 and tot["errors"] == 0

    # ---- 3. fail-closed (one virtual rank's dispatcher stopped) ------------------------------------------------
    tier.tiers[V - 1].stop()
    new_inputs()
    t0 = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0
    err1 = tier.error()
    t0 = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    t_second = time.perf_counter() - t0
    tier.stop()
    tier.clear_error()
    tier.restart_from_device()
    q, s, i_np, w_np = new_inputs()
    graph.replay()
    bad3, nm3 = verify(q, s, i_np, w_np, "after restart")
    print(f"3. fail-closed: virtual rank {V - 1}'s dispatcher stopped -> replay took {t_first * 1e3:.0f} ms (timeout "
          f"{args.timeout_ms:.0f} ms), error word {err1}; second replay {t_second * 1e3:.1f} ms (short-circuit); restarted: "
          f"{nm3} missing rows, {bad3} wrong, error word {tier.error()}", flush=True)
    ok3 = err1 != 0 and t_first < args.timeout_ms / 1e3 * L + 1.0 and t_second < 0.05 and bad3 == 0 and tier.error() == 0
    tier.close()
    passed = ok1 and ok2 and ok3
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
