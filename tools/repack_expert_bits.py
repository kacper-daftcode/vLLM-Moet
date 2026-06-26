#!/usr/bin/env python3
"""Degrade FP4 (e2m1) MoE expert weights to a per-tensor K-level codebook.

Offline quality probe for a bit-plane expert-quantization scheme on
DeepSeek-V4-Flash. For every routed-expert weight tensor
(`layers.<L>.ffn.experts.<E>.{w1,w2,w3}.weight`, dtype I8, two FP4 e2m1
codes packed per byte) the tool:

  1. builds the 16-bin code histogram (both nibbles),
  2. picks K levels out of the 16 representable e2m1 values by exact
     weighted 1-D dynamic programming (cluster = contiguous run of sorted
     values, representative constrained to a representable value inside the
     run), minimizing the weighted L2 error of the value distribution;
     +0 / -0 are merged into one histogram bin for the DP,
  3. remaps every 4-bit code to the nearest chosen level's ORIGINAL e2m1
     code (sign bit preserved when the target level is zero), repacks two
     codes per byte.

Scale tensors (`*.scale`, F8_E8M0, block-32) and every non-expert tensor
pass through byte-identical; safetensors headers are preserved verbatim, so
output files are byte-compatible with the originals (same size, offsets).
With K=16 the mapping is the identity and output files must be
byte-identical to the inputs (correctness gate, see `gate` subcommand).

The MTP drafter (`mtp.*`) and shared experts (`*.shared_experts.*`) do NOT
match the filter and are never modified.

Usage:
  # roundtrip correctness gate (K=16 must be byte-identical):
  python3 tools/repack_expert_bits.py gate --src /path/to/DeepSeek-V4-Flash \
      --files model-00002-of-00046.safetensors model-00046-of-00046.safetensors

  # build a degraded variant (symlinks for untouched files):
  python3 tools/repack_expert_bits.py build --src /path/to/DeepSeek-V4-Flash \
      --dst /path/to/output-2bit --levels 4 --workers 16
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# e2m1 magnitude table for codes 0..7; codes 8..15 are the negatives
# (sign in the high bit of the 4-bit code, sign-magnitude representation).
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
CODE_VALUES = np.concatenate([E2M1_MAG, -E2M1_MAG])  # value of code 0..15

EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def file_has_experts(path):
    hdr, _ = read_header(path)
    return any(EXPERT_RE.match(k) for k in hdr)


def choose_levels(hist16, K):
    """Pick K of the 16 representable e2m1 values minimizing the weighted L2
    error of the code-value distribution. Exact via 1-D DP over the sorted
    distinct values (+0/-0 merged into a single weight bin).

    Returns (sorted level list, total weighted squared error)."""
    # merge +0 and -0 into one point; drop values that never occur
    pts = {}
    for c in range(16):
        v = float(CODE_VALUES[c]) + 0.0  # +0.0 normalizes -0.0 to 0.0
        pts[v] = pts.get(v, 0) + int(hist16[c])
    items = sorted((v, w) for v, w in pts.items() if w > 0)
    v = np.array([p[0] for p in items], dtype=np.float64)
    w = np.array([p[1] for p in items], dtype=np.float64)
    m = len(v)
    K = min(K, m)

    INF = float("inf")
    # cost[i][j]: best weighted SSE for points i..j with one representative
    # chosen among v[i..j]; rep[i][j]: index of that representative.
    cost = np.full((m, m), INF)
    rep = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        vv = v[i:]
        ww = w[i:]
        # errs[r, j] = sum_{t<=j} w_t * (v_t - v_r)^2 over the suffix slice
        d2 = (vv[None, :] - vv[:, None]) ** 2 * ww[None, :]
        csum = np.cumsum(d2, axis=1)
        for j in range(i, m):
            col = csum[: j - i + 1, j - i]
            r = int(np.argmin(col))
            cost[i][j] = float(col[r])
            rep[i][j] = i + r

    # DP over number of clusters
    D = np.full((K + 1, m), INF)
    P = np.zeros((K + 1, m), dtype=np.int64)
    D[1, :] = cost[0, :]
    for k in range(2, K + 1):
        for j in range(k - 1, m):
            cand = D[k - 1, k - 2 : j] + cost[k - 1 : j + 1, j]
            i = int(np.argmin(cand))
            D[k, j] = cand[i]
            P[k, j] = i + k - 1
    # backtrack
    levels = []
    j, k = m - 1, K
    while k >= 1:
        i = int(P[k, j]) if k > 1 else 0
        levels.append(float(v[rep[i][j]]))
        j, k = i - 1, k - 1
    levels.sort()
    return levels, float(D[K, m - 1])


def build_lut(levels):
    """16-entry code map (code -> nearest-level code) and the derived
    256-entry byte LUT (both packed nibbles remapped at once)."""
    lv = np.asarray(levels, dtype=np.float64)
    cmap = np.zeros(16, dtype=np.uint8)
    for c in range(16):
        x = float(CODE_VALUES[c])
        L = float(lv[int(np.argmin(np.abs(lv - x)))])
        if L == 0.0:
            # keep sign bit on reconstruction: +0 stays code 0, -0 stays code 8
            cmap[c] = 8 if c >= 8 else 0
        else:
            mag = int(np.nonzero(E2M1_MAG == abs(L))[0][0])
            cmap[c] = mag | (8 if L < 0.0 else 0)
    b = np.arange(256, dtype=np.uint16)
    lut = (cmap[b & 0xF] | (cmap[b >> 4] << np.uint8(4))).astype(np.uint8)
    return lut, cmap


# ---------------------------------------------------------------------------
# Granularity / symmetry modes (sub-tensor or constrained codebooks).
#
# In every mode the codebook is still K levels chosen from the 16 representable
# e2m1 values (output stays valid FP4); only the GROUP over which the
# histogram is taken -- and, for `tensor-sym`, a symmetry constraint on the
# level set -- changes vs the per-tensor baseline:
#
#   tensor      one codebook for the whole tensor  (== per (expert,tensor); the
#               POISON reference; uses the exact per-tensor DP above)
#   tensor-sym  whole tensor, levels constrained sign-SYMMETRIC: {-b,-a,a,b}
#               (two +/- mirror pairs) -- tests the diagnosed asymmetry directly
#   expert      one codebook per expert, shared across its w1/w2/w3 tensors
#   row         one codebook per output row (axis-0 of the I8 matrix)
#   block32     one codebook per 32-value block (== one F8_E8M0 scale block,
#               i.e. 16 contiguous bytes) -- the fine-granularity ceiling
#
# For K=4 the level set is found by EXHAUSTIVE search over all C(15,4)=1365
# 4-subsets of the e2m1 value grid (or the 21 sign-symmetric {+/-a,+/-b} subsets
# for tensor-sym), minimizing weighted L2 with nearest-level assignment -- the
# same objective as the per-tensor DP, evaluated independently per group and
# fully vectorised (one [G,16]x[16,nsub] matmul) so the 8.7G blocks of block32
# stay tractable. K>=15 is the identity map (used by the byte-identity gate to
# also exercise the per-group reshape/apply paths).
#
# Nearest-level distance ties are broken odd-symmetrically (sign-matching
# level first, then smallest magnitude) so the remap is an odd function of
# the weight value: first-argmin tie-breaking would send BOTH +-0 codes (and
# tied magnitudes like +-2 vs {1,3}) to the more-negative level, injecting a
# systematic negative bias -- fatal for tensor-sym, where every zero (~12% of
# mass) would land on -a. The per-tensor DP path (`build_lut`) keeps its
# original first-argmin behavior: it is the published POISON reference and
# its optimal codebooks are sign-asymmetric (no +- ties in practice).
GRANULARITIES = ("tensor", "tensor-sym", "expert", "row", "block32")

# 15 distinct e2m1 values (+0/-0 merged), sorted ascending.
GRID = np.array(sorted({float(v) + 0.0 for v in CODE_VALUES}), dtype=np.float64)


def _e2m1_code_for_level(level, orig_code):
    """e2m1 code reconstructing `level`; sign of zero copied from orig_code."""
    if level == 0.0:
        return 8 if orig_code >= 8 else 0
    mag = int(np.nonzero(E2M1_MAG == abs(level))[0][0])
    return mag | (8 if level < 0.0 else 0)


def _candidate_level_sets(K, symmetric):
    """List of candidate level sets (each a tuple of K grid values)."""
    if symmetric:
        # sign-symmetric K=4 == two +/- mirror pairs {-b,-a,a,b}; a K=4 set
        # that is closed under negation cannot contain a lone zero, so zeros
        # map sign-preservingly to the smallest chosen magnitude.
        assert K == 4, "sign-symmetric mode is defined for K=4 only"
        mags = [float(m) for m in E2M1_MAG if m > 0]
        sets = []
        for i in range(len(mags)):
            for j in range(i + 1, len(mags)):
                a, b = mags[i], mags[j]
                sets.append((-b, -a, a, b))
        return sets
    from itertools import combinations
    return [tuple(GRID[list(idx)]) for idx in combinations(range(len(GRID)), K)]


def _nearest_level(lv, c):
    """Nearest level to code c's value, distance ties broken odd-symmetrically:
    prefer the level whose sign matches the code's sign bit (zero levels are
    sign-neutral), then the smallest magnitude. Keeps the remap an odd
    function of the weight value, so symmetric weight distributions stay
    unbiased (see the tie-breaking note in the mode comment above)."""
    x = float(CODE_VALUES[c])
    d = np.abs(lv - x)
    cand = [float(L) for L in lv[d == d.min()]]
    if len(cand) > 1:
        neg = c >= 8
        match = [L for L in cand if L == 0.0 or (L < 0.0) == neg]
        if match:
            cand = match
    return min(cand, key=abs)


def _subset_tables(K, symmetric):
    """Per-candidate (err16, cmap16) tables for the vectorised selector.

    err16[s, c]  = (value(code c) - nearest level in set s)^2
    cmap16[s, c] = e2m1 code of that nearest level (sign-preserving for 0,
                   odd-symmetric tie-breaking; ties don't affect err16)."""
    sets = _candidate_level_sets(K, symmetric)
    err16 = np.zeros((len(sets), 16), dtype=np.float64)
    cmap16 = np.zeros((len(sets), 16), dtype=np.uint8)
    for s, levels in enumerate(sets):
        lv = np.asarray(levels, dtype=np.float64)
        for c in range(16):
            L = _nearest_level(lv, c)
            err16[s, c] = (float(CODE_VALUES[c]) - L) ** 2
            cmap16[s, c] = _e2m1_code_for_level(L, c)
    return err16, cmap16


def _levels_from_cmap(cmap16):
    """Distinct reconstructed levels implied by a 16-entry code map."""
    return sorted({float(CODE_VALUES[int(v)]) + 0.0 for v in cmap16})


def build_byte_lut_from_cmap(cmap16):
    """256-entry byte LUT remapping both packed nibbles via a 16-entry code map."""
    b = np.arange(256, dtype=np.uint16)
    return (cmap16[b & 0xF] | (cmap16[b >> 4] << np.uint8(4))).astype(np.uint8)


def _select_cmaps(hist, err16, cmap16):
    """hist [G,16] -> (cmap [G,16] uint8, per-group min weighted SSE [G]).

    float64 for the few-group cases (whole-tensor / per-expert counts are huge);
    float32 is exact and ~2x cheaper for the many-group row/block cases (small
    per-group counts)."""
    dt = np.float64 if hist.shape[0] <= 1024 else np.float32
    cost = hist.astype(dt) @ err16.T.astype(dt)            # [G, nsub]
    best = np.argmin(cost, axis=1)
    sse = cost[np.arange(cost.shape[0]), best]
    return cmap16[best], sse


def _group_hist(low, high):
    """Per-group 16-bin code histogram from nibble arrays [G,bpg] -> [G,16]."""
    hist = np.empty((low.shape[0], 16), dtype=np.float64)
    for c in range(16):
        hist[:, c] = (low == c).sum(1) + (high == c).sum(1)
    return hist


def _remap_groups(seg, bpg, K, err16, cmap16, chunk=1 << 17):
    """Remap a contiguous byte view, grouped into rows of `bpg` bytes, with one
    K-level codebook per group. Modifies `seg` in place; returns summed SSE.
    K>=15 applies the identity map (exercises the reshape/apply path)."""
    arr = seg.reshape(-1, bpg)
    identity = np.arange(16, dtype=np.uint8)
    sse_sum = 0.0
    for s in range(0, arr.shape[0], chunk):
        sub = arr[s : s + chunk]
        low = (sub & 0x0F).astype(np.intp)
        high = (sub >> 4).astype(np.intp)
        if K >= 15:
            cmap = np.broadcast_to(identity, (sub.shape[0], 16))
        else:
            cmap, sse = _select_cmaps(_group_hist(low, high), err16, cmap16)
            sse_sum += float(sse.sum())
        nl = np.take_along_axis(cmap, low, axis=1)
        nh = np.take_along_axis(cmap, high, axis=1)
        sub[:] = nl | (nh << np.uint8(4))
    return sse_sum


def process_file(src, dst, K, granularity="tensor"):
    """Repack one safetensors file: degrade matching expert tensors to a
    K-level codebook at the requested granularity, stream everything else
    through unchanged. Returns stats."""
    t0 = time.time()
    buf = np.fromfile(src, dtype=np.uint8)
    (n,) = struct.unpack("<Q", buf[:8].tobytes())
    hdr = json.loads(buf[8 : 8 + n].tobytes())
    base = 8 + n

    symmetric = granularity == "tensor-sym"
    err16 = cmap16 = None
    if granularity in ("tensor-sym", "expert", "row", "block32") and K < 15:
        err16, cmap16 = _subset_tables(K, symmetric)

    items = [(name, meta) for name, meta in hdr.items()
             if name != "__metadata__" and EXPERT_RE.match(name)]
    for _, meta in items:
        assert meta["dtype"] == "I8", f"unexpected dtype: {meta['dtype']}"

    n_expert = 0
    tot_w = tot_sse = tot_ssq = 0.0
    codebooks = Counter()

    if granularity == "expert":
        groups = {}
        for name, meta in items:
            m = EXPERT_RE.match(name)
            groups.setdefault((int(m.group(1)), int(m.group(2))), []).append(meta)
        for key, metas in groups.items():
            # all three mats must live in this shard, or the shared codebook
            # would silently be fit on a partial histogram
            assert len(metas) == 3, f"expert group {key}: {len(metas)} mats"
        for metas in groups.values():
            segs = [buf[base + mt["data_offsets"][0] : base + mt["data_offsets"][1]]
                    for mt in metas]
            hist16 = np.zeros(16)
            for seg in segs:
                h256 = np.bincount(seg, minlength=256).reshape(16, 16)
                hist16 += h256.sum(0) + h256.sum(1)
            tot_w += float(hist16.sum())
            tot_ssq += float((hist16 * CODE_VALUES**2).sum())
            if K < 15:
                cmap, sse = _select_cmaps(hist16[None, :], err16, cmap16)
                lut = build_byte_lut_from_cmap(cmap[0])
                for seg in segs:
                    seg[:] = lut[seg]
                tot_sse += float(sse[0])
                codebooks[tuple(_levels_from_cmap(cmap[0]))] += 1
            n_expert += len(segs)
    else:
        for name, meta in items:
            o0, o1 = meta["data_offsets"]
            seg = buf[base + o0 : base + o1]
            h256 = np.bincount(seg, minlength=256).reshape(16, 16)
            hist16 = h256.sum(0) + h256.sum(1)
            tot_w += float(hist16.sum())
            tot_ssq += float((hist16 * CODE_VALUES**2).sum())
            if granularity == "tensor":
                levels, sse = choose_levels(hist16, K)
                lut, _ = build_lut(levels)
                seg[:] = lut[seg]
                tot_sse += sse
                codebooks[tuple(levels)] += 1
            elif granularity == "tensor-sym":
                if K < 15:
                    cmap, sse = _select_cmaps(hist16[None, :], err16, cmap16)
                    seg[:] = build_byte_lut_from_cmap(cmap[0])[seg]
                    tot_sse += float(sse[0])
                    codebooks[tuple(_levels_from_cmap(cmap[0]))] += 1
            elif granularity == "row":
                tot_sse += _remap_groups(seg, meta["shape"][1], K, err16, cmap16)
            elif granularity == "block32":
                tot_sse += _remap_groups(seg, 16, K, err16, cmap16)
            n_expert += 1

    if dst is not None:
        tmp = dst + ".tmp"
        buf.tofile(tmp)
        os.replace(tmp, dst)
    return {
        "file": os.path.basename(src),
        "n_expert_tensors": n_expert,
        "rms_err": (tot_sse / tot_w) ** 0.5 if tot_w else 0.0,
        "rel_rms": (tot_sse / tot_ssq) ** 0.5 if tot_ssq else 0.0,
        "seconds": round(time.time() - t0, 1),
        "codebooks": {json.dumps(list(k)): c for k, c in codebooks.items()},
    }


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cmd_gate(args):
    """K=16 roundtrip must reproduce byte-identical files (in every mode)."""
    ok = True
    tmpdir = args.tmp or tempfile.mkdtemp(prefix="repack-gate-", dir="/dev/shm")
    os.makedirs(tmpdir, exist_ok=True)
    for fn in args.files:
        src = os.path.join(args.src, fn)
        dst = os.path.join(tmpdir, fn)
        stats = process_file(src, dst, 16, args.granularity)
        same = sha256(src) == sha256(dst)
        ok &= same
        print(f"[gate:{args.granularity}] {fn}: experts={stats['n_expert_tensors']} "
              f"byte-identical={same} ({stats['seconds']}s)")
        os.remove(dst)
    print(f"[gate:{args.granularity}] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _worker(job):
    src, dst, K, granularity = job
    return process_file(src, dst, K, granularity)


def cmd_build(args):
    src_dir, dst_dir, K = args.src, args.dst, args.levels
    os.makedirs(dst_dir, exist_ok=True)
    entries = sorted(os.listdir(src_dir))
    jobs = []
    for e in entries:
        sp = os.path.join(src_dir, e)
        dp = os.path.join(dst_dir, e)
        if e.endswith(".safetensors") and file_has_experts(sp):
            jobs.append((sp, dp, K, args.granularity))
        else:
            # symlink everything we don't rewrite (configs, tokenizer dirs,
            # index json, expert-free shards)
            if os.path.islink(dp):
                os.remove(dp)
            if not os.path.lexists(dp):
                os.symlink(sp, dp)
    print(f"[build] K={K} granularity={args.granularity}: rewriting {len(jobs)} "
          f"shard(s), symlinking {len(entries) - len(jobs)} entries -> {dst_dir}")
    t0 = time.time()
    all_stats = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for st in ex.map(_worker, jobs):
            all_stats.append(st)
            print(f"[build] {st['file']}: experts={st['n_expert_tensors']} "
                  f"rms={st['rms_err']:.4f} rel_rms={st['rel_rms']:.4f} "
                  f"({st['seconds']}s)", flush=True)
    n = sum(s["n_expert_tensors"] for s in all_stats)
    rel = (sum(s["rel_rms"] ** 2 * s["n_expert_tensors"] for s in all_stats) / max(n, 1)) ** 0.5
    books = Counter()
    for s in all_stats:
        for k, c in s["codebooks"].items():
            books[k] += c
    print(f"[build] DONE K={K} granularity={args.granularity}: {n} expert "
          f"tensors, mean rel-RMS={rel:.4f}, wall={time.time() - t0:.0f}s")
    if books:
        print("[build] top codebooks:")
        for k, c in books.most_common(8):
            print(f"    {c:6d}x {k}")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump({"K": K, "granularity": args.granularity,
                       "n_expert_tensors": n, "mean_rel_rms": rel,
                       "codebooks": dict(books), "files": all_stats}, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="K=16 byte-identity roundtrip check")
    g.add_argument("--src", required=True)
    g.add_argument("--files", nargs="+", required=True)
    g.add_argument("--tmp", default=None)
    g.add_argument("--granularity", choices=GRANULARITIES, default="tensor")
    b = sub.add_parser("build", help="build a degraded variant directory")
    b.add_argument("--src", required=True)
    b.add_argument("--dst", required=True)
    b.add_argument("--levels", type=int, required=True, choices=range(2, 17))
    b.add_argument("--granularity", choices=GRANULARITIES, default="tensor")
    b.add_argument("--workers", type=int, default=16)
    b.add_argument("--stats-out", default=None)
    args = ap.parse_args()
    return cmd_gate(args) if args.cmd == "gate" else cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
