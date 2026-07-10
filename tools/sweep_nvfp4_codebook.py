#!/usr/bin/env python3
"""Sign-asymmetry codebook sweep on a modelopt NVFP4 MoE checkpoint.

Ports the GLM-5.2 sweep (the sign-symmetric codebook finding) to NVFP4
checkpoints (nvidia/Kimi-K2.7-Code-NVFP4, nvidia/GLM-5.2-NVFP4): for a
sample of routed-expert tensors it

  1. dequantizes NVFP4 (e2m1 codes x e4m3 block-16 scale x f32 scale_2)
     to f64 — exact, all factors representable,
  2. re-quantizes through the load-time pipeline of moe_w2_planes
     (block-32 UE8M0 scale -> e2m1 snap) to get the unit-space e2m1
     nibble histogram the 2-bit codebook then quantizes,
  3. per tensor, picks (a) the optimal-L2 4-level codebook by the exact
     per-tensor DP of repack_expert_bits (first-argmin tie-break — the
     POISON reference; its optimum is sign-ASYMMETRIC in practice) and
     (b) the best sign-SYMMETRIC {-b,-a,a,b} set (odd tie-break),
  4. reports the weighted mean signed error (bias) and rel-RMS of both.

The DS4/GLM-5.2 finding this must reproduce before shipping a new model:
the asym optimum drops one sign's tail -> a one-signed bias that compounds
over layers (GLM-5.2: -0.042 mean, 99% negative; sym ~392x smaller at
equal rel-RMS), and the winning symmetric set is {-4,-1,1,4}.

Usage:
  python3 tools/sweep_nvfp4_codebook.py --src /root/models/Kimi-K2.7-Code-NVFP4 \
      --layers 1,8,15,22,30,38,46,53,60 --experts 0,47,96,191,288,383 \
      [--prefix language_model.model] [--device cuda]
"""

import argparse
import importlib.util
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_expert_bits import (  # noqa: E402
    CODE_VALUES,
    _subset_tables,
    choose_levels,
    build_lut,
)


def _load_planes_module():
    """moe_w2_planes without importing the vllm package (mirrors the golden
    test): prefer an installed patched vllm, else the local patch worktree."""
    cands = []
    try:
        spec = importlib.util.find_spec("vllm")
        if spec and spec.submodule_search_locations:
            cands.append(os.path.join(
                spec.submodule_search_locations[0],
                "model_executor/layers/quantization/utils/moe_w2_planes.py"))
    except Exception:  # noqa: BLE001
        pass
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../vllm-v0240/vllm/model_executor/layers/"
                              "quantization/utils/moe_w2_planes.py"))
    for p in cands:
        if os.path.exists(p):
            s = importlib.util.spec_from_file_location("moe_w2_planes", p)
            m = importlib.util.module_from_spec(s)
            s.loader.exec_module(m)
            return m
    raise SystemExit("moe_w2_planes.py not found (install patched vllm or "
                     "keep a patched worktree next to vLLM-Moet)")


_planes = _load_planes_module()


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def nibble_hist(src, prefix, layer, expert, proj, device):
    """Unit-space e2m1 nibble histogram [16] of one expert tensor after the
    load-time requant (NVFP4 f64 dequant -> block-32 UE8M0 -> e2m1 snap)."""
    from safetensors import safe_open
    base = f"{prefix}.layers.{layer}.mlp.experts.{expert}.{proj}"
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    names = {sfx: f"{base}.{sfx}"
             for sfx in ("weight", "weight_scale", "weight_scale_2")}
    tensors = {}
    by_file = {}
    for sfx, name in names.items():
        by_file.setdefault(wm[name], []).append((sfx, name))
    for fname, items in by_file.items():
        with safe_open(os.path.join(src, fname), framework="pt") as f:
            for sfx, name in items:
                tensors[sfx] = f.get_tensor(name)

    w = tensors["weight"].to(device)                  # [N, K/2] u8
    s = tensors["weight_scale"].to(device)            # [N, K/16] e4m3
    s2 = tensors["weight_scale_2"].to(device)         # scalar f32
    group = w.shape[1] * 2 // s.shape[1]
    _, _, nib = _planes.nvfp4_to_codes_scales(
        w, s, s2, group=group, want_nibbles=True)
    return torch.bincount(nib.flatten().long(), minlength=16).cpu().numpy()


def codebook_stats(hist16, err16_sym, cmap16_sym):
    """Per-tensor stats for the asym-optimal DP codebook and the best
    sign-symmetric set. bias/rel-RMS are histogram-weighted in unit space.

    The sym set is scored twice: `bias_sym` with the validated
    sign-preserving zero map (+0 -> +a, -0 -> -a), and `bias_sym_alt` with
    one-signed zeros rebalanced 50/50 across +-a — the loader's
    zero-sign-balancing policy for one-signed-zero NVFP4 exports
    (Kimi-K2.7; see moe_w2_planes._f64_to_codes_scales). L2 is identical
    by construction, only the signed error changes."""
    vals = CODE_VALUES.copy()
    vals[8] = -0.0
    tot = hist16.sum()
    rms = np.sqrt((hist16 * vals ** 2).sum() / tot)

    def _stats_from_cmap(cmap, split_zeros=False):
        lv = CODE_VALUES[cmap.astype(int)]
        err = lv - vals
        h = hist16.astype(np.float64).copy()
        if split_zeros:
            # rebalance exact zeros 50/50 between the +-a images
            z = h[0] + h[8]
            e0, e8 = err[0], err[8]
            bias_z = z / 2 * (e0 + e8)
            bias_nz = (np.delete(h, [0, 8]) * np.delete(err, [0, 8])).sum()
            bias = (bias_z + bias_nz) / tot
        else:
            bias = (h * err).sum() / tot
        rel = np.sqrt((h * err ** 2).sum() / tot) / max(rms, 1e-30)
        return bias, rel

    # (a) optimal-L2 asymmetric: exact DP + first-argmin LUT (POISON ref)
    levels_a, _ = choose_levels(hist16, 4)
    _, cmap_a = build_lut(levels_a)
    bias_a, rel_a = _stats_from_cmap(cmap_a)
    sym_a = sorted(levels_a) == sorted(-l for l in levels_a)

    # (b) best sign-symmetric set (odd tie-break tables)
    cost = hist16.astype(np.float64) @ err16_sym.T
    s_best = int(np.argmin(cost))
    bias_s, rel_s = _stats_from_cmap(cmap16_sym[s_best])
    bias_sa, _ = _stats_from_cmap(cmap16_sym[s_best], split_zeros=True)
    lv_s = sorted(set(float(CODE_VALUES[int(c)]) + 0.0
                      for c in cmap16_sym[s_best]))
    zero_frac = float((hist16[0] + hist16[8]) / tot)
    zero_pos_frac = float(hist16[0] / max(hist16[0] + hist16[8], 1))
    return dict(levels_asym=levels_a, asym_is_sym=sym_a,
                bias_asym=bias_a, rel_asym=rel_a,
                levels_sym=lv_s, bias_sym=bias_s, bias_sym_alt=bias_sa,
                rel_sym=rel_s, zero_frac=zero_frac,
                zero_pos_frac=zero_pos_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prefix", default="language_model.model")
    ap.add_argument("--layers", default="1,8,15,22,30,38,46,53,60")
    ap.add_argument("--experts", default="0,47,96,191,288,383")
    ap.add_argument("--projs", default="gate_proj,up_proj,down_proj")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    err16_sym, cmap16_sym = _subset_tables(4, symmetric=True)
    layers = [int(x) for x in args.layers.split(",")]
    experts = [int(x) for x in args.experts.split(",")]
    projs = args.projs.split(",")

    rows = []
    for L in layers:
        for E in experts:
            for P in projs:
                hist = nibble_hist(args.src, args.prefix, L, E, P, args.device)
                st = codebook_stats(hist, err16_sym, cmap16_sym)
                st.update(layer=L, expert=E, proj=P)
                rows.append(st)
        print(f"layer {L}: {len(experts) * len(projs)} tensors done",
              flush=True)

    n = len(rows)
    ba = np.array([r["bias_asym"] for r in rows])
    bs = np.array([r["bias_sym"] for r in rows])
    bsa = np.array([r["bias_sym_alt"] for r in rows])
    ra = np.array([r["rel_asym"] for r in rows])
    rs = np.array([r["rel_sym"] for r in rows])
    zf = np.array([r["zero_frac"] for r in rows])
    zpf = np.array([r["zero_pos_frac"] for r in rows])
    asym_actually_sym = sum(r["asym_is_sym"] for r in rows)
    sym14 = sum(r["levels_sym"] == [-4.0, -1.0, 1.0, 4.0] for r in rows)

    print(f"\n=== sweep: {n} tensors ===")
    print(f"zeros: {zf.mean() * 100:.2f}% of mass, "
          f"{zpf.mean() * 100:.1f}% encoded +0 "
          f"({'ONE-SIGNED export' if zpf.mean() > 0.95 or zpf.mean() < 0.05 else 'balanced'})")
    print(f"asym-optimal DP:   mean bias {ba.mean():+.5f}  "
          f"(negative on {np.mean(ba < 0) * 100:.0f}%, "
          f"mean |bias| {np.abs(ba).mean():.5f}), mean rel-RMS {ra.mean():.4f}")
    print(f"  (DP optimum sign-symmetric on {asym_actually_sym}/{n} tensors)")
    print(f"sign-sym, sign-preserving zeros: mean bias {bs.mean():+.5f} "
          f"(mean |bias| {np.abs(bs).mean():.5f}), mean rel-RMS {rs.mean():.4f}")
    print(f"sign-sym, zero-balanced (loader policy): mean bias {bsa.mean():+.5f} "
          f"(mean |bias| {np.abs(bsa).mean():.5f}), same rel-RMS")
    print(f"  {{-4,-1,1,4}} wins the sym search on {sym14}/{n} tensors")
    print(f"  rel-RMS penalty sym vs asym: "
          f"{(rs.mean() / ra.mean() - 1) * 100:+.1f}%")

    # Gate: the serving pipeline (sym codebook + zero-sign balancing) must
    # carry (1) negligible signed bias in absolute terms and vs the asym
    # reference, (2) the canonical {-4,-1,1,4} set winning, (3) a bounded
    # L2 penalty vs the per-tensor optimum.
    verdict = (np.abs(bsa).mean() < 0.005
               and np.abs(bsa).mean() * 5 < np.abs(ba).mean()
               and sym14 > n * 0.9
               and rs.mean() / ra.mean() < 1.05)
    print(f"\nGATE {'PASS' if verdict else 'FAIL'}: sign-symmetric codebook "
          f"+ zero policy {'holds' if verdict else 'does NOT hold'} on this model")
    out = os.environ.get("SWEEP_JSON")
    if out:
        json.dump(rows, open(out, "w"), default=float)
        print(f"rows -> {out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
