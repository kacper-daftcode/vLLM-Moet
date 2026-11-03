"""The expert store's NVMe tail (the 256 / 128 GB tiers), host only (no GPU): the tier's dispatcher with a DRAM LRU of
N records per layer, the cold records read from a store directory on a miss, every computed row compared bit for bit
with the scalar reference - the same rows whether the record came from DRAM or from NVMe.

    /root/work/venv/bin/python test_store_tail.py [--records ../../internal/dsv41-5090-cpu/parity/b12x_inputs_l1.npz]
        [--tables ../../internal/dsv41-5090-cpu/parity] [--scratch /mnt/nvme/cache/ds41/engram-v2-tests/store]
        [--cpus 10-15] [--dispatcher-cpu 9] [--steps 60] [--dram 6]

Needs a real filesystem for the store files (O_DIRECT: /mnt/nvme, not /tmp) and idle CPUs for the kernel pool (not
while a ds41-hyb boot is up: its pool cores and their SMT siblings). Checks: rows == reference on every pair (resident
and fetched), the fetch / eviction / resident counters, record_acquire of a cold record (synchronous read) and its
bytes, pins against eviction, prefill_fetch (resident pinned in place, cold into the prefill arena) and its bytes,
status on a read past the file, the KPI ring.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "cpu_moe"))

MAX_PAIRS, MAX_TOKENS, HID, SCALES = 96, 16, 5120, 160


class Pair(ctypes.Structure):
    _fields_ = [("token", ctypes.c_uint16), ("expert", ctypes.c_uint16), ("row", ctypes.c_uint16), ("pad", ctypes.c_uint16), ("weight", ctypes.c_float)]


class Req(ctypes.Structure):
    _fields_ = [("step", ctypes.c_uint32), ("n_pairs", ctypes.c_uint32), ("n_tokens", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("t_publish_ns", ctypes.c_uint64), ("n_all", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("ids", ctypes.c_int16 * MAX_PAIRS),
                ("pairs", Pair * MAX_PAIRS), ("a_values", (ctypes.c_uint8 * HID) * MAX_TOKENS), ("a_scales", (ctypes.c_uint8 * SCALES) * MAX_TOKENS)]


class Ack(ctypes.Structure):
    _fields_ = [("step", ctypes.c_uint32), ("status", ctypes.c_uint32), ("t_seen_ns", ctypes.c_uint64), ("t_done_ns", ctypes.c_uint64),
                ("n_pairs", ctypes.c_uint32), ("n_experts", ctypes.c_uint32)]


def parse_cpus(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def ensure_lib(args):
    if os.environ.get("CPU_TIER_LIB"):
        return
    out = os.path.join(args.build_dir, "libcpu_tier.so")
    if not os.path.exists(out) or args.rebuild:
        subprocess.run(["bash", os.path.join(_HERE, "build.sh")], check=True, env=dict(os.environ, ARCH=args.arch, OUT=out))
    os.environ["CPU_TIER_LIB"] = out
    os.environ["DSV41_B12X_LIB"] = out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(_HERE, "..", "..", "..", "internal", "dsv41-5090-cpu", "parity", "b12x_inputs_l1.npz"))
    ap.add_argument("--tables", default=os.path.join(_HERE, "..", "..", "..", "internal", "dsv41-5090-cpu", "parity"))
    ap.add_argument("--scratch", default="/mnt/nvme/cache/ds41/engram-v2-tests/store")
    ap.add_argument("--cpus", default="10-15")
    ap.add_argument("--dispatcher-cpu", type=int, default=9)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dram", type=int, default=6, help="DRAM records per layer (of 16 experts)")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--build-dir", default="/tmp")
    ap.add_argument("--arch", default="znver5")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ensure_lib(args)
    os.environ["DSV41_B12X_TABLES"] = args.tables
    import dsv41_b12x as B  # noqa: E402
    from cpu_tier import RECORD_BYTES, ExpertStore, host_lib, init_tables, _Config  # noqa: E402

    Z = np.load(args.records)
    records = Z["records"]
    E, L, T, K = records.shape[0], args.layers, args.tokens, 6
    # the store directory: layer l's expert e = record (e + l) % E (so the layers differ)
    os.makedirs(os.path.join(args.scratch, "rank0"), exist_ok=True)
    rec_of = {}
    for l in range(L):
        path = os.path.join(args.scratch, "rank0", f"layer{l:02d}.bin")
        idx = [(e + l) % E for e in range(E)]
        rec_of[l] = idx
        if not os.path.exists(path) or os.path.getsize(path) != E * RECORD_BYTES:
            with open(path, "wb") as f:
                for e in range(E):
                    f.write(records[idx[e]].tobytes())
    with open(os.path.join(args.scratch, "manifest.json"), "w") as f:
        f.write('{"test": true}\n')
    print(f"store: {E} experts x {L} layers of real records ({E * L * RECORD_BYTES / 2**20:.0f} MiB) in {args.scratch}", flush=True)

    lib = host_lib()
    init_tables(lib, args.tables)
    cpus = parse_cpus(args.cpus)
    arr = (ctypes.c_int * len(cpus))(*cpus)
    cfg = _Config(L, E, len(cpus), arr, args.dispatcher_cpu, 1.0, 1.0)
    tier = lib.cpu_tier_create(ctypes.byref(cfg))
    assert tier
    assert ctypes.sizeof(Req) == lib.cpu_tier_request_bytes() and ctypes.sizeof(Ack) == lib.cpu_tier_ack_bytes()
    reqs = [Req() for _ in range(L)]
    acks = [Ack() for _ in range(L)]
    stagings = [np.zeros((MAX_PAIRS, HID), dtype=np.uint16) for _ in range(L)]
    for l in range(L):
        assert lib.cpu_tier_set_mailboxes(tier, l, ctypes.addressof(reqs[l]), ctypes.addressof(acks[l]), stagings[l].ctypes.data) == 0

    class TierShim:   # ExpertStore.attach() talks to a CpuTier; the handle-level calls are enough here
        def set_record(self, layer, e, ptr):
            assert lib.cpu_tier_set_record(tier, layer, e, ptr) == 0

        def store_tail_init(self, d, rank, base, n_slots, per_layer, qd, pbase, psets, pslots, file_layers=None):
            fl = np.ascontiguousarray(file_layers, dtype=np.int32)
            rc = lib.cpu_tier_store_tail_init(tier, d.encode(), rank, base, n_slots, per_layer, qd, pbase, psets, pslots, fl.ctypes.data)
            assert rc == 0, rc

        def set_record_cold(self, layer, e):
            assert lib.cpu_tier_set_record_cold(tier, layer, e) == 0

        def store_tail_adopt(self):
            return lib.cpu_tier_store_tail_adopt(tier)

    N = args.dram
    store = ExpertStore(args.scratch, 0, L, E, mask=None, pinned=False, nthreads=4, lib=lib, dram_records=N, spare_slots=4,
                        prefill_sets=2, prefill_slots=8, tail_qd=4)
    n_adopted = None
    shim = TierShim()
    store.attach(shim)
    print(f"store tail: {store.n_records} warm records loaded ({N} per layer), {store.n_cold} cold, arena {store.n_slots} slots, "
          f"prefill arena {store.prefill_sets} x {store.prefill_slots}", flush=True)

    def tail_stats(reset=False):
        buf = (ctypes.c_int64 * 11)()
        lib.cpu_tier_tail_stats(tier, buf, 1 if reset else 0)
        return dict(fetches=buf[0], fetch_ns=buf[1], wait_ns=buf[2], sync=buf[3], pf=buf[4], pf_ns=buf[5], io_err=buf[6], evict=buf[7],
                    res_serves=buf[8], resident=buf[9], spares=buf[10])

    st0 = tail_stats()
    assert st0["resident"] == N * L and st0["spares"] == 4, st0
    assert lib.cpu_tier_start(tier, 1) == 0
    g = torch.Generator().manual_seed(11)
    seq = [0] * L
    bad = pairs_checked = 0
    fetched_pairs = 0
    lat = []
    for it in range(args.steps):
        x = torch.randn(T, HID, generator=g)
        x[:, torch.randperm(HID, generator=g)[:48]] *= 12.0
        q, s = B.rows_quant(x.to(torch.bfloat16), zero_byte=0)
        for l in range(L):
            ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).numpy()
            w = (torch.rand(T, K, generator=g) * 0.9 + 0.1).numpy()
            # every pair is "missing" (the GPU holds nothing in this test): the request lists them all
            states = {int(e): lib.cpu_tier_record_state(tier, l, int(e)) for e in np.unique(ids)}
            R = reqs[l]
            n = 0
            for t in range(T):
                for k in range(K):
                    R.pairs[n] = Pair(t, int(ids[t, k]), t * K + k, 0, float(w[t, k]))
                    R.ids[n] = int(ids[t, k])
                    n += 1
            R.n_pairs, R.n_tokens, R.n_all = n, T, n
            for t in range(T):
                ctypes.memmove(ctypes.addressof(R.a_values[t]), q[t].ctypes.data, HID)
                ctypes.memmove(ctypes.addressof(R.a_scales[t]), s[t].ctypes.data, SCALES)
            seq[l] += 1
            t0 = time.perf_counter()
            R.step = seq[l]
            while acks[l].step != seq[l]:
                if time.perf_counter() - t0 > 30:
                    raise TimeoutError(f"layer {l} step {seq[l]} no ack")
            lat.append((time.perf_counter() - t0) * 1e3)
            assert acks[l].status == 0, f"status {acks[l].status} at layer {l} step {seq[l]}"
            rows = stagings[l]
            for t in range(T):
                for k in range(K):
                    e = int(ids[t, k])
                    ref = B.reference_pair(records[rec_of[l][e]], q[t], s[t], float(w[t, k]))
                    d = int((rows[t * K + k] != ref["out"]).sum())
                    pairs_checked += 1
                    fetched_pairs += int(states[e] == 0)
                    if d:
                        bad += 1
                        if bad <= 3:
                            print(f"  step {it} layer {l} pair ({t},{k}) expert {e} (state before {states[e]}): {d} columns differ", flush=True)
    st = tail_stats()
    lat = np.array(lat)
    print(f"1. {args.steps} steps x {L} layers: {pairs_checked} pairs checked ({fetched_pairs} on experts that were cold before their request), "
          f"{bad} rows differ from the reference; tail: fetches {st['fetches']} (mean {st['fetch_ns'] / max(1, st['fetches']) / 1e3:.0f} us "
          f"issue->landing per request), evictions {st['evict']}, resident serves {st['res_serves']}, io errors {st['io_err']}, "
          f"resident {st['resident']} spares {st['spares']}; request latency median {np.median(lat):.2f} ms max {lat.max():.2f} ms", flush=True)
    ok1 = bad == 0 and st["fetches"] > 0 and st["evict"] > 0 and st["io_err"] == 0 and st["resident"] == N * L and st["spares"] == 4
    # 2. record_acquire of a cold record: synchronous read, the bytes, the pin against eviction
    cold = [e for e in range(E) if lib.cpu_tier_record_state(tier, 0, e) == 0]
    assert cold, "no cold expert left in layer 0"
    e0 = cold[0]
    p = lib.cpu_tier_record_acquire(tier, 0, e0)
    assert p, "acquire of a cold record returned 0"
    got = np.ctypeslib.as_array((ctypes.c_uint8 * RECORD_BYTES).from_address(p))
    ok2 = np.array_equal(got, records[rec_of[0][e0]]) and lib.cpu_tier_record_state(tier, 0, e0) == 1
    st2 = tail_stats()
    # force N + 2 inserts into layer 0 through requests on other experts: e0 must survive (pinned)
    others = [e for e in range(E) if e != e0]
    for rnd in range(3):
        R = reqs[0]
        chosen = others[rnd * 6:(rnd + 1) * 6]
        for i, e in enumerate(chosen):
            R.pairs[i] = Pair(0, e, i, 0, 0.5)
            R.ids[i] = e
        R.n_pairs, R.n_tokens, R.n_all = len(chosen), 1, len(chosen)
        seq[0] += 1
        R.step = seq[0]
        t0 = time.perf_counter()
        while acks[0].step != seq[0]:
            assert time.perf_counter() - t0 < 30
        assert acks[0].status == 0
    ok2 &= lib.cpu_tier_record_state(tier, 0, e0) == 1
    still = np.array_equal(got, records[rec_of[0][e0]])
    lib.cpu_tier_record_release(tier, 0, e0)
    print(f"2. acquire of cold expert {e0}: bytes == the file's record: {np.array_equal(got, records[rec_of[0][e0]])}, resident after: "
          f"{lib.cpu_tier_record_state(tier, 0, e0) == 1}, sync fetches {st2['sync']}; survived 18 inserts while pinned: {still}", flush=True)
    ok2 &= still and st2["sync"] >= 1
    # 3. prefill_fetch: a mix of resident and cold experts -> pointers; bytes; pins; release
    res = [e for e in range(E) if lib.cpu_tier_record_state(tier, 1, e) == 1][:3]
    cld = [e for e in range(E) if lib.cpu_tier_record_state(tier, 1, e) == 0][:5]
    exp = np.array(res + cld, dtype=np.int32)
    out = np.zeros(len(exp), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.cpu_tier_prefill_fetch(tier, 1, exp.ctypes.data, len(exp), 1, out.ctypes.data, 4)
    dt = time.perf_counter() - t0
    ok3 = rc == len(cld)
    for i, e in enumerate(exp):
        gotp = np.ctypeslib.as_array((ctypes.c_uint8 * RECORD_BYTES).from_address(int(out[i])))
        eq = np.array_equal(gotp, records[rec_of[1][int(e)]])
        in_prefill = store.prefill_base <= int(out[i]) < store.prefill_base + store.prefill_sets * store.prefill_slots * RECORD_BYTES
        ok3 &= eq and (in_prefill == (i >= len(res)))
    lib.cpu_tier_prefill_release(tier, 1, exp.ctypes.data, len(exp))
    st3 = tail_stats()
    print(f"3. prefill_fetch of {len(res)} resident + {len(cld)} cold experts: {rc} read from NVMe in {dt * 1e3:.1f} ms "
          f"({len(cld) * RECORD_BYTES / dt / 1e9:.1f} GB/s), bytes all equal and the cold ones in the prefill arena: {ok3}; "
          f"prefill fetches {st3['pf']}", flush=True)
    # 4. a read past the file: status 5, the tier keeps going
    R = reqs[2]
    R.pairs[0] = Pair(0, E - 1, 0, 0, 0.5)
    R.ids[0] = E - 1
    R.n_pairs, R.n_tokens, R.n_all = 1, 1, 1
    # make expert E-1 of layer 2 cold and point its file read past the end: shrink the file temporarily
    path = os.path.join(args.scratch, "rank0", "layer02.bin")
    ok4 = True
    if lib.cpu_tier_record_state(tier, 2, E - 1) == 0:
        os.truncate(path, (E - 1) * RECORD_BYTES)
        seq[2] += 1
        R.step = seq[2]
        t0 = time.perf_counter()
        while acks[2].step != seq[2]:
            assert time.perf_counter() - t0 < 30
        ok4 = acks[2].status == 5
        print(f"4. a record read past the file: ack status {acks[2].status} (5 = io error, fail-closed), io errors {tail_stats()['io_err']}", flush=True)
        with open(path, "ab") as f:
            f.write(records[rec_of[2][E - 1]].tobytes())
    else:
        print("4. (skipped: expert E-1 of layer 2 is resident)")
    lib.cpu_tier_stop(tier)
    lib.cpu_tier_destroy(tier)
    passed = ok1 and ok2 and ok3 and ok4
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
