#!/usr/bin/env python3
"""Prefill lever bench for moe_w2_mm: validate + benchmark the AFRAG variant.

PROFILE FINDING (ncu, RTX 5090 / SM120, realistic prefill, M=16, contig experts):
  moe_w2_mm is L1/LOAD-ISSUE bound, NOT weight-DRAM bound.
    K=4096 (w13): L1/TEX 91.6% | DRAM 17.3% | L2 hit 89% | Compute(SM) 36.6%
    K=2048 (w2) : L1/TEX 83.4% | DRAM 16.2% | L2 hit 88% | Compute(SM) 37.5%
  Even worst-case (scattered experts) DRAM tops out ~59% -- weight is L2-resident.
  Dominant stall = Long Scoreboard (LDG latency); occupancy reg-limited 4 CTA/SM.
  Per-warp LDG mix ~117/k-loop: A 55%, scales 27%, codes 7%, As 7%.

LEVER (this file's variant): AFRAG -- store the activation A FRAGMENT-MAJOR so each
  lane's whole m16k32 QMMA A-fragment (4 words) is contiguous and loads in ONE
  LDG.128. That collapses the 8 strided 4-byte A loads/k64 down to 2, cutting the
  dominant load class ~4x at IDENTICAL occupancy (regcount unchanged -> 4 CTA/SM).
  QMMA inputs are bit-identical to MC=4, so numerics are unchanged.
  (The hypothesized M-blocking lever -- MB=2, 32-token tile -- was implemented and
   measured too: it amortizes decode but doubles accumulator regs -> 2 CTA/SM ->
   latency-bound -> 1.6-1.8x SLOWER. See MOEW2_MB=2; reported as a negative.)

Quality gate: every pair's bf16 vs f32 ref rel <= 3.4e-3, deterministic across RUNS,
and bit-exact vs the current MC=4 cubin. Then perf vs MC=4 at K in {2048,4096}.

Run: CUDA_VISIBLE_DEVICES=4 python3 tools/moe_w2_prefill_bench.py
Env: E, PAIRS, RUNS, SEED, KS (csv of K), GPU note: GPU 4 only.
"""
import ast
import ctypes
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/cubit/tools")
from culaunch import Cuda  # noqa: E402


def _import_packers():
    """Reuse the EXACT pack_fragment_major / pack_scales from moe_w2_check.py
    WITHOUT executing its script body (it runs a validation + sys.exit on import).
    """
    src = open("/workspace/cubit/tools/moe_w2_check.py").read()
    ns = {"torch": torch, "np": np}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "pack_fragment_major", "pack_scales"):
            exec(compile(ast.Module([node], []), "moe_w2_check.py", "exec"), ns)
    return ns["pack_fragment_major"], ns["pack_scales"]


pack_fragment_major, pack_scales = _import_packers()

CUBIT = "/workspace/cubit/target/release/cubit"
GEN = "/workspace/cubit/tools/gen_moe_w2.py"
STUB = "/workspace/cubit/sass/qmma_e4m3.merc.stub"
SCRATCH = "/tmp"                       # scratch cubins only -- never /tmp/cubit-share
PROD_MC4 = "/tmp/cubit-share/moe_w2_mm_mc4_k{K}.cubin"   # read-only baseline

N = int(os.environ.get("N", "4096"))
E = int(os.environ.get("E", "8"))                 # experts (validation pairs)
PAIRS = int(os.environ.get("PAIRS", "1024"))      # benchmark token-groups
RUNS = int(os.environ.get("RUNS", "4"))
KS = [int(x) for x in os.environ.get("KS", "2048,4096").split(",")]
torch.manual_seed(int(os.environ.get("SEED", "7")))
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0])
# per-expert real token counts (NOT multiples of 16) -> partial last tiles, exactly
# like moe_align_block_size at prefill. With extra=4 over-alloc, slots%16 == 4 (the
# real 120k-context crash had pairs=848, slots=13572 = 848*16 + 4).
RAGGED_COUNTS = [int(x) for x in os.environ.get(
    "RAGGED_COUNTS", "50,17,9,32,23,5,41,28,13,60").split(",")]


def pack_a_fragment_major(a8: torch.Tensor) -> torch.Tensor:
    """[16, K] fp8 bytes -> fragment-major: per lane (g,t) per global k64, the 32
    bytes are the lane's m16k32 A-fragment for k32a (16B) then k32b (16B), i.e. the
    4 QMMA words {row g q0, row g+8 q0, row g q1, row g+8 q1} contiguous per k32.

    Layout dims  [g2, g, j, quad, t, b]  (row=g2*8+g, k=64j+16*quad+4t+b)
    -> permute   [j,  g, t, quad, g2, b]  -> flatten.  Byte offset of lane (g,t)
    in k64 j = (4g+t)*32; the kernel reads base + wid*16*KSLICE + lane*32 + 1024*j.
    """
    M, K = a8.shape
    assert M == 16, "AFRAG tile is m16"
    a = a8.view(2, 8, K // 64, 4, 4, 4)            # g2, g, j, quad, t, b
    a = a.permute(2, 1, 4, 3, 0, 5).contiguous()   # j, g, t, quad, g2, b
    return a.reshape(-1)


def _frag_vllm(a: torch.Tensor, pairs: int, K: int) -> torch.Tensor:
    """EXACT mirror of vllm moe_w2_cubit._to_fragment_major (the FIXED helper):
    it requires `a` to have EXACTLY pairs*16 rows. The integration BUG was calling
    this on ws['a1'][:slots], where slots = sorted_ids.numel() is moe_align's
    OVER-ALLOCATED size (topk*T + E*15) and NOT a multiple of 16 -> reshape size
    mismatch -> hard crash. The fix slices the tile-aligned region [:pairs*16]."""
    v = a.view(torch.uint8).view(pairs, 2, 8, K // 64, 4, 4, 4)
    v = v.permute(0, 3, 2, 5, 4, 1, 6).reshape(pairs * 16, K)
    return v.contiguous().view(a.dtype)


def build_ragged(K, counts):
    """Mimic the REAL moe_align layout: per-expert token counts that are NOT
    multiples of 16 -> partial last tiles with within-tile ZERO filler (gathered
    pad row). Returns experts, per-tile (expert, nreal, a8[16,K], as[16,K//128],
    ref[16,N]). num_post = pairs*16; the caller adds trailing over-alloc filler."""
    experts, tiles = [], []
    for n in counts:
        codes = torch.randint(0, 4, (N, K), dtype=torch.uint8)
        sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
        w_deq = LEVELS[codes.long()] * torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
        eidx = len(experts)
        experts.append(dict(codes=codes, sexp=sexp))
        a = torch.randn(n, K) * 0.5
        ab = a.view(n, K // 128, 128)
        a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
        a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, K)
        ref = (a8.float() * a_s.float().repeat_interleave(128, 1)) @ w_deq.T   # [n, N]
        for t in range((n + 15) // 16):
            r0, r1 = t * 16, min(t * 16 + 16, n)
            nreal = r1 - r0
            a8t = torch.zeros(16, K, dtype=torch.uint8)
            a8t[:nreal] = a8[r0:r1].view(torch.uint8)
            ast = torch.zeros(16, K // 128, dtype=torch.float32)
            ast[:nreal] = a_s[r0:r1].float()
            reft = torch.zeros(16, N, dtype=torch.float32)
            reft[:nreal] = ref[r0:r1]
            tiles.append(dict(e=eidx, nreal=nreal, a8=a8t, asc=ast, ref=reft))
    return experts, tiles


def validate_ragged(cu, base, afrag, K, counts, extra=4):
    """Reproduce the ragged-tile integration crash, then validate the fix.
    slots = pairs*16 + extra (extra makes slots NOT a multiple of 16, exactly like
    sorted_ids over-allocation -> the '4 leftover tokens' of the real crash)."""
    experts, tiles = build_ragged(K, counts)
    pairs = len(tiles)
    n_af = pairs * 16
    slots = n_af + extra

    # (1) reproduce the ORIGINAL crash: the vLLM helper on a[:slots] (slots%16!=0).
    reproduced = False
    try:
        _frag_vllm(torch.zeros(slots, K, dtype=torch.uint8), pairs, K)
    except RuntimeError:
        reproduced = True

    a1_row = torch.cat([t["a8"] for t in tiles], 0)        # [n_af, K] row-major
    asc = torch.cat([t["asc"] for t in tiles], 0)          # [n_af, K//128]
    refs = torch.cat([t["ref"] for t in tiles], 0)         # [n_af, N]
    d_b = [cu.to_device(pack_fragment_major(e["codes"]).numpy()) for e in experts]
    d_bs = [cu.to_device(pack_scales(e["sexp"]).numpy()) for e in experts]
    d_as = cu.to_device(asc.numpy().astype(np.float32).view(np.uint8))
    d_c = cu.alloc(n_af * N * 2)

    def run(cubin, a_host):
        fn = cu.load_kernel(cubin, "moe_w2_mm")
        d_a = cu.to_device(a_host.contiguous().numpy())
        descs = np.zeros((pairs, 6), dtype=np.uint64)
        for p, t in enumerate(tiles):
            descs[p] = [d_a.value + p * 16 * K, d_as.value + p * 16 * (K // 128) * 4,
                        d_b[t["e"]].value, d_bs[t["e"]].value,
                        d_c.value + p * 16 * N * 2, 16]
        d_desc = cu.to_device(descs.view(np.uint8))
        args = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64),
                ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
        outs = []
        for _ in range(RUNS):
            cu.memset32(d_c, 0, n_af * N // 2)
            cu.launch(fn, (N // 16, pairs, 1), (256, 1, 1), args)
            cu.synchronize()
            outs.append(cu.from_device(d_c, n_af * N * 2, dtype=np.uint16).copy().tobytes())
        cu.free(d_a)
        cu.free(d_desc)
        return outs

    out_mc4 = run(base, a1_row)
    # FIX: repack EXACTLY the tile-aligned region (n_af = pairs*16 rows), as the
    # patched call site does (ws['a1'][:pairs*16]); a1_row already has n_af rows.
    out_af = run(afrag, _frag_vllm(a1_row, pairs, K))
    for d in d_b + d_bs + [d_as, d_c]:
        cu.free(d)

    det = len(set(out_af)) == 1
    bitexact = out_mc4[0] == out_af[0]
    realmask = torch.zeros(n_af, dtype=torch.bool)
    off = 0
    for t in tiles:
        realmask[off:off + t["nreal"]] = True
        off += 16

    def _rel(blob):
        raw = np.frombuffer(blob, dtype=np.uint16).copy()
        got = torch.from_numpy(raw).view(torch.bfloat16).float().reshape(n_af, N)
        return ((got[realmask] - refs[realmask]).abs().max()
                / refs[realmask].abs().max()).item()

    return dict(pairs=pairs, slots=slots, reproduced=reproduced,
                rel=_rel(out_af[0]), rel_mc4=_rel(out_mc4[0]),
                det=det, bitexact=bitexact)


def asm(mc, afrag, mb, K, tag):
    sass = f"{SCRATCH}/moe_prefill_{tag}_k{K}.sass"
    cubin = f"{SCRATCH}/moe_prefill_{tag}_k{K}.cubin"
    env = dict(os.environ, MOEW2_MC=str(mc), MOEW2_AFRAG=str(afrag),
               MOEW2_MB=str(mb))
    subprocess.run([sys.executable, GEN, sass, str(K)], check=True, env=env,
                   capture_output=True)
    r = subprocess.run([CUBIT, "asm", sass, "-o", cubin, "--kernel", "moe_w2_mm",
                        "--mercury-stub", STUB], capture_output=True, text=True)
    out = r.stdout + r.stderr
    if "0 failed" not in out or r.returncode != 0:
        raise RuntimeError(f"asm {tag} k{K} FAILED:\n{out[-1500:]}")
    return cubin


def build_expert(K):
    codes = torch.randint(0, 4, (N, K), dtype=torch.uint8)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
    a = torch.randn(16, K) * 0.5
    ab = a.view(16, K // 128, 128)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(16, K)
    w_deq = LEVELS[codes.long()] * torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    ref = (a8.float() * a_s.float().repeat_interleave(128, 1)) @ w_deq.T   # [16, N]
    return dict(codes=codes, sexp=sexp, a8=a8, a_s=a_s, ref=ref)


def validate(cu, cubin, K, afrag, experts):
    """One launch over the SHARED E (expert,tile) pairs, M=16; check each vs f32
    ref + determinism across RUNS. Returns (worst_rel, distinct, host_out)."""
    fn = cu.load_kernel(cubin, "moe_w2_mm")
    descs = np.zeros((E, 6), dtype=np.uint64)
    d_cs, keep = [], []
    for ex in experts:
        a_bytes = (pack_a_fragment_major(ex["a8"].view(torch.uint8)) if afrag
                   else ex["a8"].view(torch.uint8).reshape(-1))
        d_a = cu.to_device(a_bytes.numpy())
        d_as = cu.to_device(ex["a_s"].float().numpy().astype(np.float32).view(np.uint8))
        d_b = cu.to_device(pack_fragment_major(ex["codes"]).numpy())
        d_bs = cu.to_device(pack_scales(ex["sexp"]).numpy())
        d_c = cu.alloc(16 * N * 2)
        d_cs.append(d_c)
        keep += [d_a, d_as, d_b, d_bs]
        descs[len(d_cs) - 1] = [d_a.value, d_as.value, d_b.value, d_bs.value,
                                d_c.value, 16]
    d_desc = cu.to_device(descs.view(np.uint8))
    args = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64),
            ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
    outs, worst = [], 0.0
    for _ in range(RUNS):
        for d_c in d_cs:
            cu.memset32(d_c, 0, 16 * N // 2)
        cu.launch(fn, (N // 16, E, 1), (256, 1, 1), args)
        cu.synchronize()
        blob = b""
        for e, d_c in enumerate(d_cs):
            raw = cu.from_device(d_c, 16 * N * 2, dtype=np.uint16).copy()
            blob += raw.tobytes()
            got = torch.from_numpy(raw.reshape(16, N).copy()).view(torch.bfloat16).float()
            worst = max(worst, (got - experts[e]["ref"]).abs().max().item()
                        / experts[e]["ref"].abs().max().item())
        outs.append(blob)
    for d in keep + d_cs + [d_desc]:
        cu.free(d)
    return worst, len(set(outs)), outs[0]


def bench(cu, cubin, K, afrag, pairs):
    """Time `pairs` token-groups, experts laid out CONTIGUOUSLY (realistic
    moe_align ordering). M=16 per pair."""
    fn = cu.load_kernel(cubin, "moe_w2_mm")
    planes = cu.alloc(E * N * K // 4)
    scales = cu.alloc(E * N * K // 32)
    cu.memset32(scales, 0x7f7f7f7f, E * N * K // 32 // 4)
    a = cu.alloc(pairs * 16 * K)
    as_ = cu.alloc(pairs * 16 * (K // 128) * 4)
    c = cu.alloc(pairs * 16 * N * 2)
    ppe = (pairs + E - 1) // E
    descs = np.zeros((pairs, 6), dtype=np.uint64)
    for p in range(pairs):
        e = (p // ppe) % E
        descs[p] = [a.value + p * 16 * K, as_.value + p * 16 * (K // 128) * 4,
                    planes.value + e * (N * K // 4), scales.value + e * (N * K // 32),
                    c.value + p * 16 * N * 2, 16]
    d_desc = cu.to_device(descs.view(np.uint8))
    args = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64),
            ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
    ms = cu.time_launches(fn, (N // 16, pairs, 1), (256, 1, 1), args,
                          iters=50, warmup=20)
    for d in (planes, scales, a, as_, c, d_desc):
        cu.free(d)
    return ms


def main():
    print(f"# moe_w2 prefill lever bench (GPU4/SM120)  N={N} E={E} "
          f"PAIRS={PAIRS} RUNS={RUNS}")
    cu = Cuda()
    overall_ok = True
    for K in KS:
        print(f"\n===================== K={K} "
              f"({'w13/gate-up' if K == 4096 else 'w2/down'}) =====================")
        base = asm(4, 0, 1, K, "base")             # MC=4 reference (== prod cubin)
        afrag = asm(4, 1, 1, K, "afrag")           # the AFRAG variant (scratch)

        # ---- correctness on SHARED random data: AFRAG + MC4 each vs the same f32
        #      ref; AFRAG must be bit-exact vs MC4 (identical QMMA inputs).
        experts = [build_expert(K) for _ in range(E)]
        rb, db, ob = validate(cu, base, K, False, experts)
        ra, da, oa = validate(cu, afrag, K, True, experts)
        bitexact = (ob == oa)
        # bit-exact vs the accepted MC4 cubin => identical numerics => rel==MC4 rel,
        # which is the 3.4e-3-calibrated baseline. Gate on det + bit-exact + rel.
        ok = (da == 1) and bitexact and (ra <= max(3.4e-3, rb + 1e-9))
        overall_ok &= ok
        print(f"  correctness:  MC4 rel={rb:.3e}  AFRAG rel={ra:.3e} "
              f"(thr 3.4e-3) det={da == 1} bit-exact-vs-MC4={bitexact} "
              f"-> {'PASS' if ok else 'FAIL'}")

        # ---- RAGGED prefill: real moe_align layout (slots NOT a multiple of 16,
        #      partial last tiles w/ zero filler). Reproduces the e2e crash, then
        #      confirms the tile-aligned repack fix.
        rg = validate_ragged(cu, base, afrag, K, RAGGED_COUNTS, extra=4)
        ok_rg = (rg["reproduced"] and rg["det"] and rg["bitexact"]
                 and rg["rel"] <= max(3.4e-3, rg["rel_mc4"] + 1e-9))
        overall_ok &= ok_rg
        print(f"  ragged:  pairs={rg['pairs']} slots={rg['slots']} "
              f"(slots%16={rg['slots'] % 16}, partial tiles) "
              f"old-repack-crash-reproduced={rg['reproduced']} | "
              f"AFRAG rel={rg['rel']:.3e} det={rg['det']} "
              f"bit-exact-vs-MC4={rg['bitexact']} -> {'PASS' if ok_rg else 'FAIL'}")

        # ---- perf: AFRAG vs MC4 at prefill shape (contig experts)
        ms_base = bench(cu, base, K, False, PAIRS)
        ms_af = bench(cu, afrag, K, True, PAIRS)
        print(f"  perf @{PAIRS} pairs (16 tok ea): MC4 {ms_base:.3f} ms | "
              f"AFRAG {ms_af:.3f} ms | speedup {ms_base / ms_af:.2f}x")
    cu.close()
    print(f"\nRESULT: {'PASS' if overall_ok else 'FAIL'}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
