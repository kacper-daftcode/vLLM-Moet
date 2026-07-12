#!/usr/bin/env python3
"""Nested-codebook numerics for the split FP4 kernel (moe_w4s_mm).

Facts to establish on REAL data (GLM packs hold base 2-bit codes and FP4
nibbles of the SAME experts):
  1. the invariant code == _NIBBLE_TO_CODE[nibble] holds pack-wide (it
     should by construction; broken only by the zero-sign 'alt' mode);
  2. nibble histogram per class -> choose the merged magnitude pair in the
     5-member classes (+-{0,.5,1,1.5,2});
  3. reconstruction error of the merge on real distributions (element %
     hit + unit-space L2/bias delta vs true FP4).

Encoding design being validated:
  sign     = base code (0,1 -> negative; 2,3 -> positive)
  class    = |code| in {1, 4}: small = codes 1,2 ; big = codes 0,3
  ref (2b) = index into the class's magnitude list
  small class mags {0,.5,1,1.5,2}: 5 -> merge one adjacent pair -> 4
  big   class mags {3,4,6}:        3 -> direct (1 spare)
"""
import json

import numpy as np

PACKS = "/workspace/moet-serve/packs-glm"
RANK = "rank0of2"

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, dtype=np.float64)
E2M1[8:] *= -1
NIBBLE_TO_CODE = np.array([2] * 5 + [3] * 3 + [1] * 5 + [0] * 3,
                          dtype=np.uint8)

base_meta = json.load(open(f"{PACKS}/base.{RANK}.json"))
fp4_meta = json.load(open(f"{PACKS}/fp4.{RANK}.json"))
E = base_meta["E"]
bs, fs = base_meta["slot_bytes"], fp4_meta["slot_bytes"]
bstride, fstride = base_meta["stride"], fp4_meta["stride"]
# slot = [c13|s13|c2|s2]; fp4 = [2*c13|s13|2*c2|s2]; s = c/8; c13 = 2*c2
c2 = 8 * bs // 27
c13 = 2 * c2
s13, s2 = c13 // 8, c2 // 8
assert c13 + s13 + c2 + s2 == bs, (c13, s13, c2, s2, bs)
assert 2 * c13 + s13 + 2 * c2 + s2 == fs, (fs,)
print(f"sections: c13={c13} s13={s13} c2={c2} s2={s2}")

layers = sorted(set(base_meta["layers"]) & set(fp4_meta["layers"]))
rng = np.random.default_rng(7)
pairs = [(int(li), int(ei)) for li in rng.choice(layers, 6, replace=False)
         for ei in rng.choice(E, 4, replace=False)]

fb = open(f"{PACKS}/base.{RANK}.pack", "rb")
ff = open(f"{PACKS}/fp4.{RANK}.pack", "rb")


def unpack2(b):  # packed 2-bit LE fields -> u8 array
    b = np.frombuffer(b, dtype=np.uint8)
    out = np.empty(b.size * 4, dtype=np.uint8)
    out[0::4] = b & 3
    out[1::4] = (b >> 2) & 3
    out[2::4] = (b >> 4) & 3
    out[3::4] = (b >> 6) & 3
    return out


def unpack4(b):  # packed nibbles LE -> u8 array
    b = np.frombuffer(b, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b & 15
    out[1::2] = b >> 4
    return out


hist = np.zeros(16, dtype=np.int64)
n_match = n_tot = 0
mismatch_examples = []
for li, ei in pairs:
    fb.seek(li * E * bstride + ei * bstride)
    brow = fb.read(bs)
    ff.seek(li * E * fstride + ei * fstride)
    frow = ff.read(fs)
    for boff, foff, clen in ((0, 0, c13), (c13 + s13, 2 * c13 + s13, c2)):
        codes = unpack2(brow[boff:boff + clen])
        nibs = unpack4(frow[foff:foff + 2 * clen])
        assert codes.size == nibs.size
        pred = NIBBLE_TO_CODE[nibs]
        m = int((pred == codes).sum())
        n_match += m
        n_tot += codes.size
        if m != codes.size and len(mismatch_examples) < 3:
            bad = np.nonzero(pred != codes)[0][:5]
            mismatch_examples.append(
                (li, ei, [(int(i), int(nibs[i]), int(codes[i])) for i in bad]))
    hist += np.bincount(nibs, minlength=16)  # last section's contribution

print(f"invariant code==NIBBLE_TO_CODE[nibble]: {n_match}/{n_tot} "
      f"({100.0 * n_match / n_tot:.4f}%)")
if mismatch_examples:
    print("mismatches:", mismatch_examples)

# full histogram over all sampled rows (recount both sections cleanly)
hist = np.zeros(16, dtype=np.int64)
for li, ei in pairs:
    ff.seek(li * E * fstride + ei * fstride)
    frow = ff.read(fs)
    for foff, clen in ((0, c13), (2 * c13 + s13, c2)):
        nibs = unpack4(frow[foff:foff + 2 * clen])
        hist += np.bincount(nibs, minlength=16)

tot = hist.sum()
print("\nnibble histogram (% of elements):")
for i in range(16):
    print(f"  {E2M1[i]:+5.1f}: {100.0 * hist[i] / tot:6.3f}%")

# merge candidates in the small class (mags idx 0..4 = {0,.5,1,1.5,2}):
# merging (a,b) -> keep the more frequent value; cost = moved mass * |a-b|
print("\nmerge candidates (small class), unit-space cost per element:")
for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
    ca = hist[a] + hist[8 + a]
    cb = hist[b] + hist[8 + b]
    moved = min(ca, cb)
    cost = moved * abs(E2M1[a] - E2M1[b])
    keep = a if ca >= cb else b
    print(f"  merge mags ({E2M1[a]}, {E2M1[b]}) -> keep {E2M1[keep]}: "
          f"moved {100.0 * moved / tot:.3f}% of elems, "
          f"avg unit err {cost / tot:.5f}")

# reconstruction quality of the best merge vs true FP4, element-space
best = None
for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
    ca, cb = hist[a] + hist[8 + a], hist[b] + hist[8 + b]
    cost = min(ca, cb) * abs(E2M1[a] - E2M1[b])
    if best is None or cost < best[0]:
        best = (cost, a, b, a if ca >= cb else b)
_, ma, mb, keep = best
print(f"\nchosen merge: ({E2M1[ma]}, {E2M1[mb]}) -> {E2M1[keep]}")

# build the (code, ref) tables implied by the merge
small_mags = [m for m in range(5) if m != (mb if keep == ma else ma)]
big_mags = [5, 6, 7]
print(f"small-class ref map (r->mag): {[E2M1[m] for m in small_mags]}")
print(f"big-class   ref map (r->mag): {[E2M1[m] for m in big_mags]} + spare")

# error stats: for merged-away magnitude, error = |a-b| * 2^(sexp-127) in
# real space; relative to the 2-BIT alternative the FP4 tier replaces
merged_away = mb if keep == ma else ma
frac = (hist[merged_away] + hist[8 + merged_away]) / tot
base_err = abs(E2M1[merged_away] - 1.0)      # 2-bit would snap it to +-1
new_err = abs(E2M1[merged_away] - E2M1[keep])
print(f"\nmerged-away mag {E2M1[merged_away]}: {100 * frac:.3f}% of elems; "
      f"unit err {new_err} (2-bit base would give {base_err}) -> still "
      f"{base_err / max(new_err, 1e-9):.1f}x better than base")
