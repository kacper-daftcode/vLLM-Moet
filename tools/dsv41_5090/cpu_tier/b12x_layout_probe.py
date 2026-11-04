"""b12x's prepared weight layout for the compact W4A8 path as explicit permutations, so that a promoted expert's
checkpoint-layout record can be repacked into the resident pool by a gather (phase 2.2), bit for bit what
prepare_weights produces. Runs in the serving image on one sm_120 GPU:

    python b12x_layout_probe.py --out /parity          # writes b12x_layout_{w13,w13s,w2,w2s}.npy + b12x_layout.json

Method: the prepared tensors the compact kernel reads (w13, w13_scales, w2, w2_scales, captured through a spy on
launch_w4a8_compact_micro) are a fixed rearrangement of the input nibbles / scale bytes per expert. Bit-plane
inputs (nibble i = bit b of i, b = 0..21) read the source index of every output nibble; three random inputs then
verify out == in[idx] exactly (all 16 codes; -0 codes are canonicalized to +0 by vLLM before b12x sees them and are
excluded from the check). Two experts confirm the per-expert stride.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

K, N, E_PROBE, TOPK = 5120, 576, 2, 6
dev = torch.device("cuda", 0)

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl, _preparation  # noqa: E402
import b12x.moe._shared.kernels.w4a8_compact_micro as cm  # noqa: E402

_orig_bind = _impl.TPMoEScratchPlan.bind
_compact = {}


def bind_with_compact(self, **kw):
    b = _orig_bind(self, **kw)
    hit = _compact.get(id(self))
    if hit is None or hit[0] is not self:
        hit = _compact[id(self)] = (self, _preparation._compact_launches(self.launch_plan, self.caps))
    return dataclasses.replace(b, compact_launches=hit[1]) if hit[1] is not None else b


_impl.TPMoEScratchPlan.bind = bind_with_compact
got = {}
_orig_launch = cm.launch_w4a8_compact_micro


def spy(**kw):
    got.update(kw)
    return _orig_launch(**kw)


cm.launch_w4a8_compact_micro = spy

SHAPES = dict(w13=(E_PROBE, 2 * N, K // 2), w13s=(E_PROBE, 2 * N, K // 32), w2=(E_PROBE, K, N // 2), w2s=(E_PROBE, K, N // 32))


def prepared_from(w13: torch.Tensor, w13s: torch.Tensor, w2: torch.Tensor, w2s: torch.Tensor) -> dict:
    """prepare + bind + one tiny run -> the prepared tensors the compact kernel reads, as flat uint8 on the host."""
    got.clear()
    plan_w = fused_moe.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                    params_dtype=torch.bfloat16, num_experts=E_PROBE, hidden_size=K, intermediate_size=N,
                                    w13_layout="w31")
    one = torch.ones(E_PROBE, dtype=torch.float32, device=dev)
    prepared = fused_moe.prepare_weights(plan=plan_w, w1_fp4=w13, w1_blockscale=w13s, w1_global_scale=one, a1_gscale=one,
                                         w2_fp4=w2, w2_blockscale=w2s, w2_global_scale=one, a2_gscale=one,
                                         params_dtype=torch.bfloat16)
    caps = fused_moe.Caps(max_tokens=1, num_topk=TOPK, device=dev, weight_plan=prepared.plan, core_token_counts=(1,),
                          route_num_experts=0, quant_mode="w4a8_mx", apply_router_weight_on_input=False, swiglu_limit=10.0,
                          swiglu_alpha=None, swiglu_beta=None, frozen=True)
    plan = fused_moe.plan(caps)
    (spec,) = plan.scratch_specs()
    scratch = torch.zeros(int(spec.shape[0]), dtype=torch.uint8, device=dev)
    x = torch.zeros(1, K, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(1, TOPK, dtype=torch.int32, device=dev)
    wts = torch.ones(1, TOPK, dtype=torch.float32, device=dev)
    out = torch.empty(1, K, dtype=torch.bfloat16, device=dev)
    b = fused_moe.bind(plan, scratch=scratch, a=x, experts=prepared, topk_weights=wts, topk_ids=ids, output=out,
                       input_scales_static=True, unit_scale_contract=False)
    fused_moe.run(binding=b)
    torch.cuda.synchronize()
    res = {}
    for name, key in (("w13", "w13"), ("w13s", "w13_scales"), ("w2", "w2"), ("w2s", "w2_scales")):
        t = got[key]
        res[name] = t.contiguous().view(torch.uint8).reshape(-1).cpu().numpy().copy()
        res[name + "_meta"] = (tuple(t.shape), str(t.dtype))
    return res


def nibbles(b: np.ndarray) -> np.ndarray:
    return np.stack((b & 0xF, b >> 4), axis=-1).reshape(-1)


def pack(nib: np.ndarray) -> np.ndarray:
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()
    # per-expert element counts
    n_in = dict(w13=2 * N * K // 2 * 2, w13s=2 * N * K // 32, w2=K * N // 2 * 2, w2s=K * N // 32)  # nibbles / bytes per expert
    bits = {k: int(np.ceil(np.log2(v))) for k, v in n_in.items()}
    planes = {k: [] for k in n_in}
    sizes = None
    for b in range(max(bits.values())):
        ins = {}
        for name in ("w13", "w2"):
            if b < bits[name]:
                idx = np.arange(n_in[name], dtype=np.int64)
                nib = ((idx >> b) & 1).astype(np.uint8)
                per = pack(nib)
            else:
                per = np.zeros(n_in[name] // 2, dtype=np.uint8)
            ins[name] = np.stack([per] * E_PROBE).reshape(SHAPES[name])
        for name in ("w13s", "w2s"):
            if b < bits[name]:
                idx = np.arange(n_in[name], dtype=np.int64)
                per = ((idx >> b) & 1).astype(np.uint8)
            else:
                per = np.zeros(n_in[name], dtype=np.uint8)
            ins[name] = np.stack([per] * E_PROBE).reshape(SHAPES[name])
        res = prepared_from(*(torch.from_numpy(ins[k]).to(dev) for k in ("w13", "w13s", "w2", "w2s")))
        if sizes is None:
            sizes = {k: res[k].size for k in ("w13", "w13s", "w2", "w2s")}
            print("prepared tensors:", {k: res[k + "_meta"] for k in sizes}, "bytes", sizes, flush=True)
            for k in sizes:
                exp_bytes = n_in[k] // 2 if k in ("w13", "w2") else n_in[k]
                assert sizes[k] == E_PROBE * exp_bytes, (k, sizes[k], E_PROBE * exp_bytes)
        for name in ("w13", "w2"):
            if b < bits[name]:
                planes[name].append(nibbles(res[name]))
        for name in ("w13s", "w2s"):
            if b < bits[name]:
                planes[name].append(res[name])
        print(f"  bit {b} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    idx = {}
    per_expert_out = {}
    for name in ("w13", "w13s", "w2", "w2s"):
        pl = np.stack(planes[name]).astype(np.int64)  # [bits, out elements of both experts]
        full = (pl << np.arange(pl.shape[0])[:, None]).sum(axis=0)
        n_out = full.size // E_PROBE
        per_expert_out[name] = n_out
        assert n_out == n_in[name], (name, n_out, n_in[name])
        # expert e's output block [e*n_out, (e+1)*n_out) must index expert e's input [0, n_in) (same per-expert map)
        for e in range(E_PROBE):
            blk = full[e * n_out:(e + 1) * n_out]
            assert blk.min() >= 0 and blk.max() < n_in[name], (name, e, blk.min(), blk.max())
            if e == 0:
                idx[name] = blk.astype(np.int32)
            else:
                assert np.array_equal(blk, idx[name]), f"{name}: expert {e} has a different map than expert 0"
        assert np.unique(idx[name]).size == n_in[name], f"{name}: not a permutation"
    print("maps derived; verifying on random inputs", flush=True)
    rng = np.random.default_rng(1)
    for trial in range(3):
        ins = {}
        for name in ("w13", "w2"):
            nib = rng.integers(0, 16, size=E_PROBE * n_in[name]).astype(np.uint8)
            nib[nib == 8] = 0  # -0 is canonicalized upstream
            ins[name] = pack(nib).reshape(SHAPES[name])
        for name in ("w13s", "w2s"):
            ins[name] = rng.integers(100, 140, size=E_PROBE * n_in[name]).astype(np.uint8).reshape(SHAPES[name])
        res = prepared_from(*(torch.from_numpy(ins[k]).to(dev) for k in ("w13", "w13s", "w2", "w2s")))
        for name in ("w13", "w2"):
            src = nibbles(ins[name].reshape(E_PROBE, -1)).reshape(E_PROBE, -1)
            out = nibbles(res[name]).reshape(E_PROBE, -1)
            pred = np.stack([src[e][idx[name]] for e in range(E_PROBE)])
            d = int((pred != out).sum())
            print(f"  trial {trial} {name}: {d} of {out.size} nibbles differ from the gather", flush=True)
            assert d == 0
        for name in ("w13s", "w2s"):
            src = ins[name].reshape(E_PROBE, -1)
            out = res[name].reshape(E_PROBE, -1)
            pred = np.stack([src[e][idx[name]] for e in range(E_PROBE)])
            d = int((pred != out).sum())
            print(f"  trial {trial} {name}: {d} of {out.size} bytes differ from the gather", flush=True)
            assert d == 0
    os.makedirs(args.out, exist_ok=True)
    for name in idx:
        np.save(os.path.join(args.out, f"b12x_layout_{name}.npy"), idx[name])
    meta = dict(hidden=K, inter=N, source_format="fp4_e8m0_k32", w13_layout="w31",
                prepared={k: dict(shape_for_2_experts=list(res[k + "_meta"][0]), dtype=res[k + "_meta"][1],
                                  per_expert_elements=per_expert_out[k]) for k in idx},
                note="idx[name][j] = source nibble (w13, w2) / byte (w13s, w2s) index within the expert's input "
                     "(w13 input = [w1 (gate); w3 (up)] rows, nibble 2i = low nibble of byte i) for output element j")
    with open(os.path.join(args.out, "b12x_layout.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}/b12x_layout_*.npy and b12x_layout.json in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
