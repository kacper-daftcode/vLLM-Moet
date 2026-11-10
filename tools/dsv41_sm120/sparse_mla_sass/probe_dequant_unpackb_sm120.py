#!/usr/bin/env python3
"""The dequant of sparse_mla_prefill_v42 against the dequant of sparse_mla_prefill_v41, exhaustively, on sm_120 silicon.

v42 dequantizes the staged records with the scaled BF16 unpack
(`F2FP.BF16.<E4M3|E2M1>.UNPACK_B Rd, Rs[.sel], Rscale.B0`: two codes of Rs -> two bf16 halves, each times
2^(byte 0 of Rscale - 127)); v41 unpacks to f16, widens each half to f32, packs to bf16 and multiplies by the unpacked
scale (HMUL2.BF16_V2). This script builds eight small kernels that run BOTH instruction sequences on the same inputs
and compares the results word for word. Thread (cta, tid) takes c0 = tid, c1 = cta (65 536 combinations per launch)
and the launch parameter `seed`; out[2 * gid] = the v42 sequence, out[2 * gid + 1] = the v41 sequence.

  fp8_lo     e4m3 pair (c0, c1) in bytes 0,1 (seed bits 8-23 = junk in bytes 2,3), UE8M0 scale byte = seed & 0xff
  fp8_hi     the pair in bytes 2,3 (junk in bytes 0,1), read through the source selector .H1
  fp4        e2m1 byte c0 (junk above it) x E4M3 block scale code c1: the block scale and the value by the scaled
             unpack with the unit scale code 0x7f, HMUL2 by the scale's low half (.H0_H0)
  fp4_b1/2/3 the e2m1 byte in byte 1 / 2 / 3 (junk elsewhere), read through the source's byte selector
  fp4_oldsc  the value by the scaled unpack, the block scale by v41's four instructions
  fp4_scale  the bf16 block scale alone: one scaled unpack vs v41's four instructions (low halves; the high half of
             the new word must be 0, it converts the zero byte above the code)
Sweeps: fp8_* every scale byte 0..255 (+ four more junk patterns on eight scale bytes), fp4_* eight junk seeds.
Measured on RTX PRO 6000 (sm_120): all eight kernels equal on every input — 18 874 368 results each for fp8_lo / fp8_hi,
524 288 each for the fp4 kernels; NaN codes and -0 included.

Needs: a cubit binary whose sm120 table has the instruction's row (cubit `main` at or after 89564ac), cubit's
`tools/stallfloor/culaunch.py`, numpy, one sm_120 GPU.

    probe_dequant_unpackb_sm120.py --cubit <cubit binary> --table <sm120.json(.gz)> --culaunch <dir of culaunch.py> \\
        [--out <dir>] [--gen-only | --run-only] [--device 0]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile

SAFE = "[B------:R-:W-:-:S15]"          # every line far behind its producers: this probe is about values, not stalls
NAMES = ("fp8_lo", "fp8_hi", "fp4", "fp4_b1", "fp4_b2", "fp4_b3", "fp4_oldsc", "fp4_scale")


def unpack(fmt, rd, rs, rsc, srcsel=""):
    return f"F2FP.BF16.{fmt}.UNPACK_B R{rd}, R{rs}{srcsel}, R{rsc}.B0 ;"


def v41_pair(fmt, src, scale_reg, srcsel=""):
    """v41: bf16x2 R9 = unpack(src) x scale_reg (bf16x2)."""
    return [f"F2FP.F16.{fmt}.UNPACK_B R6, R{src}{srcsel} ;", "HADD2.F32 R7, -RZ, R6.H0_H0 ;", "HADD2.F32 R8, -RZ, R6.H1_H1 ;",
            "F2FP.BF16.F32.PACK_AB R9, R8, R7 ;", f"HMUL2.BF16_V2 R9, R9, R{scale_reg} ;"]


def v41_fp4_scale(code_reg):
    """v41: bf16x2 R13 = the E4M3 block scale in both halves."""
    return [f"IMAD R13, R{code_reg}, 0x101, RZ ;", "F2FP.F16.E4M3.UNPACK_B R13, R13 ;", "HADD2.F32 R7, -RZ, R13.H0_H0 ;",
            "HADD2.F32 R13, -RZ, R13.H1_H1 ;", "F2FP.BF16.F32.PACK_AB R13, R13, R7 ;"]


def kernel(name):
    # R0 = tid (c0), R1 = cta (c1), R2 = gid, R5 = seed, R14:R15 = out
    b = ["MOV R11, 0xdeadbeef ;", "MOV R9, 0xdeadbeef ;",
         "LOP3.LUT R21, R5, 0xff, RZ, 0xc0, !PT ;",                 # seed byte 0
         "SHF.R.U32.HI R19, RZ, 0x8, R5 ;", "LOP3.LUT R19, R19, 0xffff, RZ, 0xc0, !PT ;",   # junk16 = seed bits 8-23
         "MOV R22, 0x7f ;"]                                          # the unit scale code
    new = []
    if name == "fp8_lo":
        b += ["IMAD R20, R19, 0x10000, R2 ;"]                       # junk in bytes 2,3; the pair in bytes 0,1
        new = [unpack("E4M3", 11, 20, 21)]
        b += new + ["IMAD R13, R21, 0x101, RZ ;", "F2FP.BF16.E8.UNPACK_B R13, R13 ;"] + v41_pair("E4M3", 20, 13)
    elif name == "fp8_hi":
        b += ["IMAD R20, R2, 0x10000, R19 ;"]                       # the pair in bytes 2,3; junk in bytes 0,1
        new = [unpack("E4M3", 11, 20, 21, ".H1")]
        b += new + ["IMAD R13, R21, 0x101, RZ ;", "F2FP.BF16.E8.UNPACK_B R13, R13 ;"] + v41_pair("E4M3", 20, 13, ".H1")
    elif name in ("fp4", "fp4_oldsc", "fp4_scale"):
        b += ["LOP3.LUT R20, R5, 0xffffff00, RZ, 0xc0, !PT ;", "IADD3 R20, PT, PT, R20, R0, RZ ;",   # value byte c0, junk above
              "MOV R23, R1 ;"] + v41_fp4_scale(23)                  # the scale code c1, zero-extended (a byte load)
        if name == "fp4":
            new = [unpack("E4M3", 28, 23, 22), unpack("E2M1", 11, 20, 22)]
            b += new + ["HMUL2.BF16_V2 R11, R11, R28.H0_H0 ;"] + v41_pair("E2M1", 20, 13)
        elif name == "fp4_oldsc":
            new = [unpack("E2M1", 11, 20, 22)]
            b += new + ["HMUL2.BF16_V2 R11, R11, R13 ;"] + v41_pair("E2M1", 20, 13)
        else:
            new = [unpack("E4M3", 11, 23, 22)]
            b += new + ["MOV R9, R13 ;"]
    else:
        n = int(name[-1])                                           # fp4_b<n>: the value byte in byte n
        b += [f"LOP3.LUT R20, R5, {0xffffffff ^ (0xff << (8 * n)):#x}, RZ, 0xc0, !PT ;", f"IMAD R20, R0, {1 << (8 * n):#x}, R20 ;",
              "MOV R23, R1 ;"] + v41_fp4_scale(23)
        new = [unpack("E4M3", 28, 23, 22), unpack("E2M1", 11, 20, 22, f".B{n}")]
        b += new + ["HMUL2.BF16_V2 R11, R11, R28.H0_H0 ;", f"SHF.R.U32.HI R17, RZ, {8 * n:#x}, R20 ;"] + v41_pair("E2M1", 17, 13)
    b += ["IMAD.WIDE.U32 R14, R2, 0x8, R14 ;", "STG.E desc[UR4][R14.64], R11 ;", "STG.E desc[UR4][R14.64+0x4], R9 ;",
          "STG.E desc[UR4][R14.64+0x4], R9 ;", "EXIT ;"]
    head = [f".entry {name}", "    .reg R0-R29", "    .regcount 32", "    .param ptr out", "    .param u32 seed",
            "    [B------:R-:W0:-:S04] LDCU.64 UR4, c[0x0][0x358] ;",
            "    [B------:R-:W1:-:S04] S2R R0, SR_TID.X ;",
            "    [B------:R-:W2:-:S04] S2R R1, SR_CTAID.X ;",
            "    [B------:R-:W3:-:S08] LDC.64 R14, c[0x0][0x380] ;",
            "    [B------:R-:W4:-:S08] LDC R5, c[0x0][0x388] ;",
            "    [B01234-:R-:W-:-:S15] IMAD R2, R1, 0x100, R0 ;"]
    return "\n".join(head + [f"    {SAFE} {l}" for l in b] + [".endentry", ""]), new


def generate(cubit, table, out):
    os.makedirs(out, exist_ok=True)
    idx = []
    for name in NAMES:
        src, new = kernel(name)
        with open(f"{out}/{name}.sass", "w") as f:
            f.write(src)
        r = subprocess.run([cubit, "asm", "-t", table, f"{out}/{name}.sass", "-o", f"{out}/{name}.cubin"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"cubit asm {name} failed: {(r.stderr or r.stdout)[-400:]}")
        idx.append(dict(name=name, cubin=name + ".cubin", new=new))
    with open(f"{out}/index.json", "w") as f:
        json.dump(idx, f, indent=1)
    print(f"{len(idx)} probe kernels in {out}")


JUNKS = (0xa5c3, 0x0000, 0xffff, 0x7f80, 0x1234)


def run_one(out, name, device, culaunch_dir):
    import numpy as np
    sys.path.insert(0, culaunch_dir)
    from culaunch import Cuda
    ent = [e for e in json.load(open(os.path.join(out, "index.json"))) if e["name"] == name][0]
    if name in ("fp8_lo", "fp8_hi"):
        seeds = [s | (0xa5c3 << 8) for s in range(256)] + [s | (j << 8) for j in JUNKS[1:] for s in (0, 1, 114, 118, 127, 128, 254, 255)]
        what = "65 536 e4m3 pairs x 256 scale bytes"
    else:
        seeds = [0x00000000, 0xffffffff, 0xa5c3e17b, 0x5a3c1e84, 0x7f807f80, 0x12345678, 0x80000001, 0x0f0f0f0f]
        what = "256 E4M3 scale codes" if name == "fp4_scale" else "256 e2m1 bytes x 256 E4M3 scale codes"
    cu = Cuda(device)
    fn = cu.load_kernel(os.path.join(out, ent["cubin"]), name)
    n = 65536 * 2
    gid = np.arange(65536)
    c0, c1 = gid & 0xff, gid >> 8
    tot = bad = untouched = hi_nonzero = 0
    ex = []
    for seed in seeds:
        d_out = cu.alloc(n * 4)
        cu.memset32(d_out, 0, n)
        cu.launch(fn, grid=(256, 1, 1), block=(256, 1, 1), args=[d_out, ctypes.c_uint32(seed)])
        cu.synchronize()
        got = cu.from_device(d_out, n * 4, dtype=np.uint32).reshape(-1, 2)
        cu.free(d_out)
        new, old = got[:, 0], got[:, 1]
        untouched += int(np.count_nonzero((new == 0xdeadbeef) | (old == 0xdeadbeef)))
        if name == "fp4_scale":
            hi_nonzero += int(np.count_nonzero(new >> 16))
            diff = (new & 0xffff) != (old & 0xffff)
        else:
            diff = new != old
        tot += new.size
        bad += int(np.count_nonzero(diff))
        for j in np.nonzero(diff)[0][:3]:
            if len(ex) < 12:
                ex.append(f"c0 {int(c0[j]):#04x} c1 {int(c1[j]):#04x} seed {seed:#010x}: v42 {int(new[j]):#010x} v41 {int(old[j]):#010x}")
    cu.close()
    ok = bad == 0 and untouched == 0 and hi_nonzero == 0
    print(f"{'PASS' if ok else 'FAIL'} {name} [{' '.join(ent['new'])}]: {len(seeds)} launches x 65 536 threads; v42 == v41 on {tot - bad} of {tot} "
          f"results ({what}, junk in the unused bytes)" + (f"; DIFFERENT {bad}" if bad else "")
          + (f"; untouched {untouched}" if untouched else "") + (f"; high half of the scale word non-zero in {hi_nonzero}" if hi_nonzero else ""), flush=True)
    for e in ex:
        print(f"    {e}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubit", default="cubit")
    ap.add_argument("--table")
    ap.add_argument("--culaunch", help="directory of cubit's tools/stallfloor/culaunch.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)       # one kernel in this process
    args = ap.parse_args()
    out = args.out or tempfile.mkdtemp(prefix="dequant_unpackb_")
    if args.one:
        sys.exit(0 if run_one(out, args.one, args.device, args.culaunch) else 1)
    if not args.run_only:
        if not args.table:
            ap.error("--table is needed to assemble the kernels")
        generate(args.cubit, args.table, out)
    if args.gen_only:
        return 0
    if not args.culaunch:
        ap.error("--culaunch is needed to launch the kernels")
    fails = 0
    for name in NAMES:      # one process per kernel: a driver error ends only that kernel
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--out", out, "--culaunch", args.culaunch, "--device", str(args.device), "--one", name])
        fails += r.returncode != 0
    print("ALL PASS" if fails == 0 else f"{fails} of {len(NAMES)} kernels FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
