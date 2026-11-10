#!/usr/bin/env python3
"""Bits and time of schedule variants of the hand-written MoE GEMM kernels (sm_120).

A schedule variant of a kernel is the same source and the same instructions with other control words (stalls,
barriers, pads) -- for example `cubins-sm120-s1/<stem>.cubin`, the dependency-aware stall schedule, next to the pinned
`cubins-sm120/<stem>.cubin` (see the MANIFEST, "Dependency-aware schedule set"). Bit identity on identical inputs is
therefore the gate, and the time is what the variant is for.

Inputs are built exactly as the families' own check scripts build them (moe_w2_check.py, moe_w4_check.py,
moe_w4q_check.py, moe_w4s_check.py, moe_w2_afrag_check.py: the same packers, seeds, desc ABI and launch geometry).

  default      per kernel, one process: bits -- every variant launched --runs times, the variants taking turns, on 5
               (expert, token-group) pairs x 4096 rows of the harness inputs; every output must equal the first output
               of the reference variant byte for byte; the reference's rel-vs-fp32 verdict is the sanity check.
               time -- the variants interleaved (rotating order), CUDA events, --rounds rounds of --reps launches,
               E = --e-time pairs of random planes (and E = 8 for the decode kernels, MC = 1).
  --cold P     P fresh processes per kernel: each loads every variant, launches each --cold-launches times on
               numpy-built random planes (fixed seeds) and prints the sha256 of every output; all digests of a kernel
               -- over variants, launches and processes -- must be one value. A schedule race can hide in one warm
               process and show only across processes.

    python3 kernels/gen/moe_sched_ab.py                                   # pinned vs cubins-sm120-s1/, every kernel of the set
    python3 kernels/gen/moe_sched_ab.py --cold 30 --jobs 6
    python3 kernels/gen/moe_sched_ab.py --variants pinned=,mine=.new --cubins /tmp/build --only mc4afrag

A variant is LABEL=SUFFIX (file <cubins>/<stem><SUFFIX>.cubin) or LABEL=DIR/ (file <cubins>/<DIR>/<stem>.cubin: DIR is
taken relative to <cubins>, the default `s1=../cubins-sm120-s1/` is the set's directory next to the pinned one); the
first variant is the reference. A variant whose file is byte-identical to the reference's is reported as such. Exit status 1
when any output differs, a kernel fails to load or launch, or nothing was measured. Needs an sm_120 card (the cubins
are sm_120-only), numpy, and torch on the CPU for the packers of the default mode.
"""
import argparse, collections, concurrent.futures, ctypes, glob, hashlib, json, os, re, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cubins", default=os.path.join(os.path.dirname(HERE), "cubins-sm120"))
ap.add_argument("--variants", default="pinned=,s1=../cubins-sm120-s1/",
                help="LABEL=SUFFIX or LABEL=DIR/, comma separated (file <stem><SUFFIX>.cubin or <DIR>/<stem>.cubin, DIR relative to --cubins; first = reference)")
ap.add_argument("--out", default="")
ap.add_argument("--only", default="")
ap.add_argument("--list", default="", help="file with one stem per line (order kept)")
ap.add_argument("--reps", type=int, default=25)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--e-time", type=int, default=64)
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--subprocess", action="store_true", help="one process per kernel (fault isolation)")
ap.add_argument("--limit-s", type=float, default=1e9, help="do not start another kernel after this many seconds")
ap.add_argument("--no-time", action="store_true")
ap.add_argument("--dry", action="store_true", help="CPU only: build the inputs, load nothing, launch nothing")
ap.add_argument("--cold", type=int, default=0, help="fresh processes per kernel")
ap.add_argument("--jobs", type=int, default=6, help="fresh processes at a time")
ap.add_argument("--cold-one", default="", help=argparse.SUPPRESS)
ap.add_argument("--cold-rep", type=int, default=0, help=argparse.SUPPRESS)
ap.add_argument("--cold-launches", type=int, default=2)
ap.add_argument("--cold-pairs", type=int, default=4)
ap.add_argument("--one", default="", help="stem:label - build the timing inputs and launch that variant --one-launches times (for a profiler)")
ap.add_argument("--one-launches", type=int, default=1)
args = ap.parse_args()
sys.path.insert(0, HERE)
VARIANTS = collections.OrderedDict(v.split("=", 1) for v in args.variants.split(","))
REF = next(iter(VARIANTS))

cu = drv = None
if not args.dry and not (args.cold and not args.cold_one):
    from culaunch import Cuda, _drv  # noqa: E402
    cu = Cuda()
    drv = _drv()
    drv.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    drv.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    drv.cuEventSynchronize.argtypes = [ctypes.c_void_p]
    drv.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    drv.cuMemFree_v2.argtypes = [ctypes.c_void_p]


class _DryBuf:
    value = 0


def to_device(a):
    return cu.to_device(a) if cu else _DryBuf()


def dev_alloc(n):
    return cu.alloc(n) if cu else _DryBuf()


def event():
    e = ctypes.c_void_p()
    assert drv.cuEventCreate(ctypes.byref(e), 0) == 0
    return e


def elapsed(a, b):
    ms = ctypes.c_float()
    assert drv.cuEventElapsedTime(ctypes.byref(ms), a, b) == 0
    return ms.value * 1000.0  # us


# ---------------------------------------------------------------- packers (verbatim from the family check scripts)
_T = {}


def T():
    """torch and the lookup tables of the packers, on first use (the fresh-process mode never needs them)."""
    if not _T:
        import torch
        _T["torch"] = torch
        _T["LEVELS"] = torch.tensor([-4.0, -1.0, 1.0, 4.0])
        e = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2)
        e[8:] *= -1
        _T["E2M1"] = e
        _T["NIBBLE_TO_CODE"] = torch.tensor([2] * 5 + [3] * 3 + [1] * 5 + [0] * 3, dtype=torch.uint8)
        _T["MAG_TO_REF"] = torch.tensor([0, 0, 1, 2, 3, 0, 1, 2], dtype=torch.uint8)
        _T["SMALL_VAL"] = torch.tensor([.5, 1.0, 1.5, 2.0])
        _T["BIG_VAL"] = torch.tensor([3.0, 4.0, 6.0, 6.0])
    return _T


def pack_fragment_major(codes):
    torch = T()["torch"]
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8).numpy()


def pack_fp4_fragment_major(codes):
    torch = T()["torch"]
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    c = c.view(-1, 8).to(torch.int64)
    w = (c[:, 0] | (c[:, 1] << 4) | (c[:, 2] << 8) | (c[:, 3] << 12)
         | (c[:, 4] << 16) | (c[:, 5] << 20) | (c[:, 6] << 24) | (c[:, 7] << 28))
    return w.to(torch.int32).numpy().view(np.uint8)


def pack_quintal_fragment_major(nibs):
    t = T(); torch = t["torch"]
    n, k = nibs.shape
    mag = (nibs & 7).to(torch.int64)
    code = t["NIBBLE_TO_CODE"][nibs.long()]
    big = (code == 0) | (code == 3)
    digit = torch.where(big, mag - 5, mag)
    d = digit.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    d = d.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 8, 4)
    words = (d[..., 0] + 5 * d[..., 1] + 25 * d[..., 2] + 125 * d[..., 3])
    shorts = torch.zeros(words.shape[0], 5, dtype=torch.int64)
    for w in range(8):
        b = 10 * w
        s, off = b // 16, b % 16
        shorts[:, s] |= (words[:, w] << off)
        if off > 6:
            shorts[:, s + 1] |= words[:, w] >> (16 - off)
    shorts &= 0xFFFF
    by = torch.stack([shorts & 0xFF, shorts >> 8], dim=-1).view(-1, 10)
    by = by.view(-1, 32, 10).to(torch.uint8)
    return torch.cat([by[:, :, :8].reshape(-1, 256), by[:, :, 8:].reshape(-1, 64)], dim=1).flatten().numpy()


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten().numpy()


def quant_a(a, group):
    torch = T()["torch"]
    m, k = a.shape
    ab = a.view(m, k // group, group)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    deq = a8.float() * a_s.repeat_interleave(group, 1)
    return a8, a_s, deq


def pack_a_fragment_major(a8, K):
    torch = T()["torch"]
    v = a8.view(torch.uint8).view(2, 8, K // 64, 4, 4, 4)
    v = v.permute(2, 1, 4, 3, 0, 5).reshape(16, K)
    return v.contiguous().numpy()


def nwarp_for_k(k):
    nb = k // 128
    cap = 8 if k >= 1024 else 4
    for n in range(min(cap, nb), 0, -1):
        if nb % n == 0:
            return n
    return 1


def plan_of(stem, n=None):
    fam = re.match(r"moe_(w2|w4q|w4s|w4)_mm", stem).group(1)
    K = int(re.search(r"_k(\d+)", stem).group(1)) if re.search(r"_k(\d+)", stem) else 4096
    mc = 4 if "_mc4" in stem else (2 if "_mc2" in stem else 1)
    afrag = "afrag" in stem
    a32 = stem.endswith("_a32")
    return dict(family=fam, K=K, N=n or args.n, mc=mc, afrag=afrag, a32=a32, M=4 * mc, nwarp=nwarp_for_k(K),
                kernel=f"moe_{fam}_mm", desc_words=8 if fam in ("w4q", "w4s") else 6)


def build_pair(p, seed_e, for_bits):
    """One (expert, token-group) pair: device buffers + desc row + fp32 reference (bits mode) built as the family
    harness builds them. In timing / fresh-process mode the planes are random bytes from numpy (no reference)."""
    fam, K, N, M = p["family"], p["K"], p["N"], p["M"]
    group = 32 if p["a32"] else 128
    out = {}
    if not for_bits:
        rng = np.random.default_rng(seed_e)
        a8 = rng.integers(0, 0x7e, size=(M, K), dtype=np.uint8) | (rng.integers(0, 2, size=(M, K), dtype=np.uint8) << 7)
        a_s = (rng.random((M, K // group), dtype=np.float32) * 0.01 + 0.001).astype(np.float32)
        out["a"] = to_device(a8)
        out["as"] = to_device(a_s.view(np.uint8))
        out["bs"] = to_device(rng.integers(120, 132, size=N * K // 32, dtype=np.uint8))
        if fam == "w2":
            out["b"] = to_device(rng.integers(0, 256, size=N * K // 4, dtype=np.uint8))
        elif fam == "w4":
            out["b"] = to_device(rng.integers(0, 256, size=N * K // 2, dtype=np.uint8))
        elif fam == "w4q":
            out["b"] = to_device(rng.integers(0, 256, size=N * K // 4, dtype=np.uint8))
            out["ref"] = to_device(rng.integers(0, 256, size=N * K * 5 // 16, dtype=np.uint8))
        else:  # w4s
            out["b"] = to_device(rng.integers(0, 256, size=N * K // 4, dtype=np.uint8))
            out["ref"] = to_device(rng.integers(0, 256, size=N * K // 4, dtype=np.uint8))
        out["c"] = dev_alloc(M * N * 2)
        return out, None
    t = T(); torch = t["torch"]
    torch.manual_seed(seed_e)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
    a8, a_s, a_deq = quant_a(torch.randn(M, K) * 0.5, group)
    scale = torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    if fam == "w2":
        codes = torch.randint(0, 4, (N, K), dtype=torch.uint8)
        ref = a_deq @ (t["LEVELS"][codes.long()] * scale).T
        out["b"] = to_device(pack_fragment_major(codes))
    elif fam == "w4":
        codes = torch.randint(0, 16, (N, K), dtype=torch.uint8)
        ref = a_deq @ (t["E2M1"][codes.long()] * scale).T
        out["b"] = to_device(pack_fp4_fragment_major(codes))
    elif fam == "w4q":
        nibs = torch.randint(0, 16, (N, K), dtype=torch.uint8)
        nibs.view(-1)[:16] = torch.arange(16, dtype=torch.uint8)
        code = t["NIBBLE_TO_CODE"][nibs.long()]
        ref = a_deq @ (t["E2M1"][nibs.long()] * scale).T
        out["b"] = to_device(pack_fragment_major(code))
        out["ref"] = to_device(pack_quintal_fragment_major(nibs))
    else:  # w4s (decoder's value table, merge included)
        nibs = torch.randint(0, 16, (N, K), dtype=torch.uint8)
        mag = (nibs & 7).long()
        code = t["NIBBLE_TO_CODE"][nibs.long()]
        rf = t["MAG_TO_REF"][mag]
        big = (code == 0) | (code == 3)
        val = torch.where(big, t["BIG_VAL"][rf.long()], t["SMALL_VAL"][rf.long()])
        val = torch.where((code <= 1), -val, val)
        ref = a_deq @ (val * scale).T
        out["b"] = to_device(pack_fragment_major(code))
        out["ref"] = to_device(pack_fragment_major(rf))
    out["a"] = to_device(pack_a_fragment_major(a8, K) if p["afrag"] else a8.view(torch.uint8).numpy())
    out["as"] = to_device(a_s.float().numpy().view(np.uint8))
    out["bs"] = to_device(pack_scales(sexp))
    out["c"] = dev_alloc(M * N * 2)
    return out, ref


def desc_row(p, bufs):
    M = p["M"]
    if p["desc_words"] == 6:
        return [bufs["a"].value, bufs["as"].value, bufs["b"].value, bufs["bs"].value, bufs["c"].value, np.uint64(M)]
    return [bufs["a"].value, bufs["as"].value, bufs["b"].value, bufs["ref"].value, bufs["bs"].value, bufs["c"].value, np.uint64(M), 0]


def free_all(pairs):
    if not cu:
        return
    for b in pairs:
        for v in b.values():
            drv.cuMemFree_v2(v)


def variant_path(stem, v):
    """<cubins>/<stem><SUFFIX>.cubin, or <cubins>/<DIR>/<stem>.cubin when the variant is written `LABEL=DIR/`."""
    sfx = VARIANTS[v]
    return f"{args.cubins}/{sfx}{stem}.cubin" if sfx.endswith("/") else f"{args.cubins}/{stem}{sfx}.cubin"


def load_variants(stem, p, res):
    fns = {}
    for v in VARIANTS:
        path = variant_path(stem, v)
        if not os.path.exists(path):
            continue
        try:
            fns[v] = cu.load_kernel(path, p["kernel"])
        except Exception as ex:  # noqa: BLE001
            res.setdefault("load_error", {})[v] = repr(ex)[:200]
    return fns


def launch_all(fn, grid, block, largs, pairs, M, N):
    for b in pairs:
        cu.memset32(b["c"], 0, M * N // 2)
    cu.launch(fn, grid, block, largs)
    cu.synchronize()
    return b"".join(cu.from_device(b["c"], M * N * 2).tobytes() for b in pairs)


def run_kernel(stem, p):
    K, N, M, E = p["K"], p["N"], p["M"], 5
    res = {"stem": stem, **{k: v for k, v in p.items()}}
    torch = T()["torch"]
    # ---- bits on the harness inputs
    pairs, refs = [], []
    seed0 = 7 if p["family"] == "w2" else 3
    torch.manual_seed(seed0)
    for e in range(E):
        bufs, ref = build_pair(p, seed0 * 1000 + e, True)
        pairs.append(bufs); refs.append(ref)
    if args.dry:
        res["dry"] = "inputs built"
        return res
    variants = load_variants(stem, p, res)
    res["variants"] = list(variants)
    descs = np.array([desc_row(p, b) for b in pairs], dtype=np.uint64)
    d_desc = cu.to_device(descs.view(np.uint8))
    largs = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64), ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
    grid, block = (N // 16, E, 1), (p["nwarp"] * 32, 1, 1)
    refv = REF if REF in variants else next(iter(variants))
    first = {}
    distinct = {}
    order = list(variants)
    # the variants take turns launch by launch (a schedule race shows under repetition and next to other code)
    digests = {v: [] for v in order}
    for r in range(args.runs):
        for v in order[r % len(order):] + order[:r % len(order)]:
            blob = launch_all(variants[v], grid, block, largs, pairs, M, N)
            if v not in first:
                first[v] = blob
            digests[v].append(hashlib.sha256(blob).hexdigest())
    worst = {}
    for v in order:
        w = 0.0
        for e, b in enumerate(pairs):
            raw = np.frombuffer(first[v][e * M * N * 2:(e + 1) * M * N * 2], dtype=np.uint16).copy()
            got = torch.from_numpy(raw.reshape(M, N)).view(torch.bfloat16).float()
            w = max(w, (got - refs[e]).abs().max().item() / refs[e].abs().max().item())
        worst[v] = w
        distinct[v] = len(set(digests[v]))
    refd = digests[refv][0]
    res["bits"] = {
        "reference": refv, "runs": args.runs,
        "distinct_outputs": distinct,
        "launches_equal_to_reference": {v: sum(1 for d in digests[v] if d == refd) for v in order},
        "identical": {v: all(d == refd for d in digests[v]) for v in order},
        "worst_rel": worst,
        "harness_pass": {v: w < 2.5e-2 for v, w in worst.items()},
        "sha256": {v: digests[v][0][:16] for v in order},
    }
    for v in order:
        if first[v] != first[refv]:
            a = np.frombuffer(first[refv], dtype=np.uint16); b = np.frombuffer(first[v], dtype=np.uint16)
            res["bits"].setdefault("mismatching_words", {})[v] = int((a != b).sum())
    free_all(pairs); drv.cuMemFree_v2(d_desc)
    if args.no_time:
        return res
    # ---- time: interleaved, event-timed
    shapes = [("E%d" % args.e_time, args.e_time)]
    if p["mc"] == 1:
        shapes.append(("E8", 8))
    res["time_us"] = {}
    for label, Et in shapes:
        pairs = []
        for e in range(Et):
            bufs, _ = build_pair(p, 100000 + e, False)
            pairs.append(bufs)
        descs = np.array([desc_row(p, b) for b in pairs], dtype=np.uint64)
        d_desc = cu.to_device(descs.view(np.uint8))
        largs = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64), ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
        grid = (N // 16, Et, 1)
        for v in order:  # warm-up
            for _ in range(3):
                cu.launch(variants[v], grid, block, largs)
        cu.synchronize()
        ev0, ev1 = event(), event()
        rounds = []
        for rd in range(args.rounds):
            samples = {v: [] for v in order}
            for rep in range(args.reps):
                k = (rep + rd) % len(order)
                for v in order[k:] + order[:k]:
                    drv.cuEventRecord(ev0, None)
                    cu.launch(variants[v], grid, block, largs)
                    drv.cuEventRecord(ev1, None)
                    drv.cuEventSynchronize(ev1)
                    samples[v].append(elapsed(ev0, ev1))
            rounds.append({v: float(np.median(s)) for v, s in samples.items()})
            if rd == 0:
                mins = {v: float(np.min(s)) for v, s in samples.items()}
            else:
                mins = {v: min(mins[v], float(np.min(s))) for v, s in samples.items()}
        res["time_us"][label] = {v: {"median": float(np.median([r[v] for r in rounds])), "rounds": [round(r[v], 2) for r in rounds],
                                     "min": mins[v], "n": args.reps * args.rounds} for v in order}
        free_all(pairs); drv.cuMemFree_v2(d_desc)
    return res


def cold_one(stem):
    """a fresh process: numpy-only inputs, every variant launched a few times, digests on one JSON line."""
    p = plan_of(stem, n=1024)
    K, N, M, E = p["K"], p["N"], p["M"], args.cold_pairs
    res = {"stem": stem, "rep": args.cold_rep}
    pairs = [build_pair(p, 500000 + e, False)[0] for e in range(E)]
    variants = load_variants(stem, p, res)
    descs = np.array([desc_row(p, b) for b in pairs], dtype=np.uint64)
    d_desc = cu.to_device(descs.view(np.uint8))
    largs = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64), ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
    grid, block = (N // 16, E, 1), (p["nwarp"] * 32, 1, 1)
    order = list(variants)
    k = args.cold_rep % max(len(order), 1)
    order = order[k:] + order[:k]
    dig = {v: [] for v in order}
    for _ in range(args.cold_launches):
        for v in order:
            dig[v].append(hashlib.sha256(launch_all(variants[v], grid, block, largs, pairs, M, N)).hexdigest()[:16])
    res["order"] = order
    res["digests"] = dig
    allv = {d for v in dig for d in dig[v]}
    res["all_equal"] = len(allv) == 1
    print("COLD " + json.dumps(res), flush=True)


def one_launch(spec):
    """the timing shape, one variant, a few launches: the process ncu profiles."""
    stem, v = spec.split(":")
    p = plan_of(stem)
    K, N, M, E = p["K"], p["N"], p["M"], args.e_time
    pairs = [build_pair(p, 100000 + e, False)[0] for e in range(E)]
    fn = cu.load_kernel(variant_path(stem, v), p["kernel"])
    descs = np.array([desc_row(p, b) for b in pairs], dtype=np.uint64)
    d_desc = cu.to_device(descs.view(np.uint8))
    largs = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64), ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
    for _ in range(args.one_launches):
        cu.launch(fn, (N // 16, E, 1), (p["nwarp"] * 32, 1, 1), largs)
    cu.synchronize()
    print(f"ONE {stem} {v}: {args.one_launches} launch(es), grid ({N // 16}, {E}), block {p['nwarp'] * 32}", flush=True)


def stems_to_run():
    """--list, else every kernel whose reference file and at least one other variant's file exist."""
    if args.list:
        stems = [l.strip() for l in open(args.list) if l.strip() and not l.startswith("#")]
    else:
        stems = []
        for f in sorted(glob.glob(f"{args.cubins}/moe_*{VARIANTS[REF]}.cubin")):
            stem = os.path.basename(f)[:-len(VARIANTS[REF] + ".cubin")]
            if any(v != REF and os.path.exists(variant_path(stem, v)) for v in VARIANTS):
                stems.append(stem)
    if args.only:
        stems = [s for s in stems if re.search(args.only, s)]
    return stems


def child_cmd(extra):
    return [sys.executable, os.path.abspath(__file__), "--cubins", args.cubins, "--variants", args.variants] + extra


def cold_driver(stems):
    """P fresh processes per kernel, --jobs at a time, rep-major (every kernel gets its first process before any gets
    its second); all digests of a kernel must be one value."""
    t0 = time.time()

    def one(task):
        stem, rep = task
        if time.time() - t0 > args.limit_s:
            return stem, rep, None, "skipped (time limit)"
        cmd = child_cmd(["--cold-one", stem, "--cold-rep", str(rep), "--cold-launches", str(args.cold_launches),
                         "--cold-pairs", str(args.cold_pairs)])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return stem, rep, None, "TIMEOUT"
        line = [l for l in r.stdout.splitlines() if l.startswith("COLD ")]
        if not line:
            return stem, rep, None, f"rc={r.returncode} {r.stderr.strip()[-200:]}"
        return stem, rep, json.loads(line[-1][5:]), ""

    per = {s: dict(procs=0, errors=[], digests=collections.Counter(), by_variant=collections.defaultdict(collections.Counter)) for s in stems}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for stem, rep, res, err in ex.map(one, [(s, r) for r in range(args.cold) for s in stems]):
            p = per[stem]
            if res is None:
                p["errors"].append(f"rep {rep}: {err}")
                continue
            p["procs"] += 1
            for v, ds in res["digests"].items():
                for d in ds:
                    p["digests"][d] += 1
                    p["by_variant"][v][d] += 1
            if res.get("load_error"):
                p["errors"].append(f"rep {rep}: load {res['load_error']}")
    out, npass = {}, 0
    for stem in stems:
        p = per[stem]
        ok = p["procs"] == args.cold and len(p["digests"]) == 1 and not p["errors"] and set(p["by_variant"]) == set(VARIANTS)
        npass += ok
        out[stem] = dict(processes=p["procs"], distinct_digests=len(p["digests"]), errors=p["errors"][:5],
                         by_variant={v: dict(c) for v, c in p["by_variant"].items()}, verdict="PASS" if ok else "FAIL")
        if not ok:
            print(f"COLDFAIL {stem}: {p['procs']} processes, {len(p['digests'])} distinct digests, variants {sorted(p['by_variant'])}, errors {p['errors'][:2]}", flush=True)
    if args.out:
        json.dump(dict(args=vars(args), kernels=out, seconds=time.time() - t0), open(args.out, "w"), indent=1)
    print(f"MOE_SCHED_AB fresh processes: {npass} / {len(stems)} kernels PASS ({args.cold} processes x {args.cold_launches} launches x "
          f"variants {','.join(VARIANTS)}; {time.time() - t0:.0f} s)", flush=True)
    return 0 if npass == len(stems) and stems else 1


def same_file(stem, v):
    a, b = variant_path(stem, REF), variant_path(stem, v)
    return os.path.exists(a) and os.path.exists(b) and open(a, "rb").read() == open(b, "rb").read()


def kernel_line(i, n, r):
    b, t = r.get("bits", {}), r.get("time_us", {})
    main_shape = next(iter(t.values())) if t else {}
    tm = " ".join(f"{v}={main_shape[v]['median']:.1f}" for v in main_shape) if main_shape else (r.get("error", "") or r.get("dry", ""))[:120]
    ident = b.get("identical", {})
    same = [v for v in VARIANTS if v != REF and same_file(r["stem"], v)]
    return (f"[{i + 1:2d}/{n}] {r['stem']:36s} bits ref {b.get('reference')} identical "
            + ",".join(f"{v}:{'Y' if ok else 'N'}" for v, ok in ident.items())
            + f" x{b.get('runs')} rel {max(b.get('worst_rel', {0: 0}).values()):.2e}  us {tm}"
            + (f"  ({', '.join(same)}: the reference's file)" if same else ""))


def verdict(results, skipped):
    ok = [r for r in results if "bits" in r]
    ident = [r for r in ok if all(r["bits"]["identical"].values()) and len(r["bits"]["identical"]) == len(VARIANTS)]
    line = (f"MOE_SCHED_AB bits: {len(ident)} / {len(results)} kernels identical in every variant ({','.join(VARIANTS)}; "
            f"{args.runs} warm launches each, taking turns); errors {len(results) - len(ok)}; not run {len(skipped)}")
    for v in VARIANTS:
        if v == REF:
            continue
        x, nsame = [], 0
        for r in ok:
            if same_file(r["stem"], v):
                nsame += 1
                continue
            t = next(iter(r.get("time_us", {}).values()), None)
            if t and v in t and REF in t:
                x.append(100.0 * (t[v]["median"] / t[REF]["median"] - 1.0))
        if x:
            line += (f"; time {v} vs {REF} ({len(x)} kernels with a file of their own, {nsame} with the reference's): median "
                     f"{float(np.median(x)):+.1f} % [{min(x):+.1f} .. {max(x):+.1f}], faster on {sum(1 for y in x if y < 0)} / {len(x)}")
    print(line, flush=True)
    return 0 if results and len(ident) == len(results) and not skipped else 1


def main():
    if args.cold_one:
        cold_one(args.cold_one)
        return 0
    if args.one:
        one_launch(args.one)
        return 0
    stems = stems_to_run()
    if not stems:
        print(f"MOE_SCHED_AB: no kernel with variants {','.join(VARIANTS)} under {args.cubins}")
        return 1
    if args.cold:
        return cold_driver(stems)
    results, skipped = [], []
    t0 = time.time()
    for i, stem in enumerate(stems):
        if time.time() - t0 > args.limit_s:
            skipped.append(stem)
            continue
        if args.subprocess:
            # one process per kernel: a faulting cubin (CUDA 700) must not poison the context of the others
            tmp = (args.out or os.path.join("/tmp", f"moe_sched_ab_{os.getpid()}")) + f".{stem}.part.json"
            cmd = child_cmd(["--out", tmp, "--only", f"^{re.escape(stem)}$", "--reps", str(args.reps), "--rounds", str(args.rounds),
                             "--e-time", str(args.e_time), "--runs", str(args.runs), "--n", str(args.n)] + (["--no-time"] if args.no_time else []))
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                rc, se = r.returncode, r.stderr
            except subprocess.TimeoutExpired:
                rc, se = -9, "TIMEOUT 180 s"
            if os.path.exists(tmp):
                res = json.load(open(tmp))["results"][0]
                os.remove(tmp)
            else:
                res = {"stem": stem, **plan_of(stem), "error": f"rc={rc}: {se.strip()[-300:]}"}
        else:
            p = plan_of(stem)
            try:
                res = run_kernel(stem, p)
            except Exception as ex:  # noqa: BLE001
                res = {"stem": stem, **p, "error": repr(ex)[:300]}
        results.append(res)
        print(kernel_line(i, len(stems), res) + f"  [{time.time() - t0:.0f}s]", flush=True)
        if args.out:
            json.dump({"results": results, "args": vars(args), "skipped": skipped}, open(args.out, "w"), indent=1)
    if args.out:
        json.dump({"results": results, "args": vars(args), "skipped": skipped}, open(args.out, "w"), indent=1)
    if args.dry or (len(stems) == 1 and args.only.startswith("^") and args.out.endswith(".part.json")):
        return 0          # a dry run, or the child of --subprocess: the parent prints the verdict
    return verdict(results, skipped)


if __name__ == "__main__":
    raise SystemExit(main())
