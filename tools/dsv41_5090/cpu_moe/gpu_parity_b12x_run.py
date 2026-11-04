"""GPU side 2/3 of the CPU-vs-b12x parity check: runs b12x's compact W4A8 micro MoE on the prepared inputs and dumps
every stage. Runs in the vLLM base image (FlashInfer 0.7.2 with b12x), one sm_120 GPU under its lock:

    flock -w 3600 /tmp/exl3diff-gpu1.lock docker run --rm --gpus '"device=1"' --name moet-parity-b12x \
      -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache -v <cpu_moe dir>:/work:ro -v <parity dir>:/parity \
      --entrypoint python3 vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f \
      /work/gpu_parity_b12x_run.py /parity/b12x_inputs.npz /parity/b12x_gpu.npz

b12x is driven the way vLLM's B12xExperts drives it (plan_weights w4a8_mx / fp4_e8m0_k32 / w31, prepare_weights,
Caps(max_tokens=T, top-6, swiglu_limit 10, frozen), TPMoEScratchPlan.bind + run) with the compact launchers attached
the way patch_flashinfer_b12x_compact_bind.py does (monkeypatched here; the stock bind stops at 576 rows per rank).
A spy on launch_w4a8_compact_micro keeps the scratch layout, so the dump holds: a_values / a_scales (b12x's rows
quant of the tokens), projections [pairs, 1152] bf16 (gate | up), the intermediate's e4m3 codes [pairs, 576] and
ue8m0 bytes [pairs, 18], route_output [pairs, 5120] bf16 (router weight applied), the MoE output [T, 5120] bf16,
and alpha1 / input_scale / alpha2 / down_scale. Also: the same call from a CUDA graph replayed 3 times (bits must
not move) and a gate-vs-up sanity check of the "w31" layout against fp32 math.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import torch

inp, outp = sys.argv[1], sys.argv[2]
Z = np.load(inp)
dev = torch.device("cuda", 0)
E, N2, HALF = Z["w13"].shape[0], Z["w13"].shape[1], Z["w13"].shape[2]
N = N2 // 2            # 576
K = HALF * 2           # 5120
T, TOPK = Z["ids"].shape
print(f"E={E} K={K} N={N} T={T} top-{TOPK}", flush=True)

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl, _preparation  # noqa: E402
import b12x.moe._shared.kernels.w4a8_compact_micro as cm  # noqa: E402


def u8(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev)


w13 = u8(Z["w13"]); w13s = u8(Z["w13s"]); w2 = u8(Z["w2"]); w2s = u8(Z["w2s"])
plan_w = fused_moe.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                params_dtype=torch.bfloat16, num_experts=E, hidden_size=K, intermediate_size=N,
                                w13_layout="w31")
one = torch.ones(E, dtype=torch.float32, device=dev)
prepared = fused_moe.prepare_weights(plan=plan_w, w1_fp4=w13, w1_blockscale=w13s, w1_global_scale=one, a1_gscale=one,
                                     w2_fp4=w2, w2_blockscale=w2s, w2_global_scale=one, a2_gscale=one,
                                     params_dtype=torch.bfloat16)
caps = fused_moe.Caps(max_tokens=T, num_topk=TOPK, device=dev, weight_plan=prepared.plan, core_token_counts=(T,),
                      route_num_experts=0, quant_mode="w4a8_mx", apply_router_weight_on_input=False, swiglu_limit=10.0,
                      swiglu_alpha=None, swiglu_beta=None, frozen=True)
plan = fused_moe.plan(caps)
print("launch plan:", plan.launch_plan.implementation, plan.launch_plan.quant_mode, "n", plan.launch_plan.n, flush=True)

# the compact launchers, as patch_flashinfer_b12x_compact_bind.py attaches them
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

(spec,) = plan.scratch_specs()
scratch = torch.zeros(int(spec.shape[0]), dtype=torch.uint8, device=dev)
x = torch.from_numpy(Z["x_bf16"]).view(torch.bfloat16).to(dev).contiguous()
ids = torch.from_numpy(Z["ids"]).to(dev).contiguous()
wts = torch.from_numpy(Z["weights"]).to(dev).contiguous()
out = torch.empty(T, K, dtype=torch.bfloat16, device=dev)


def run():
    b = fused_moe.bind(plan, scratch=scratch, a=x, experts=prepared, topk_weights=wts, topk_ids=ids, output=out,
                       input_scales_static=True, unit_scale_contract=False)
    fused_moe.run(binding=b)


run()
torch.cuda.synchronize()
assert got, "the compact micro path was not taken"
cap = int(got["max_tokens"])
lay = cm._layout(cap, TOPK, K, N)
st = got["scratch"].view(torch.uint8).reshape(-1)
reg = {name: st.narrow(0, *lay[name]) for name in ("a_values", "a_scales", "projections", "intermediate", "route_output")}
rows_cap = cap * TOPK
P = T * TOPK
a_values = reg["a_values"].view(cap, K)[:T].cpu().numpy()
a_scales = reg["a_scales"].view(cap, K // 32)[:T].cpu().numpy()
proj = reg["projections"].view(torch.bfloat16).view(-1, 2 * N)[:P].view(torch.int16).cpu().numpy()
wpr = 5 * 32  # words per intermediate row (640 B: five K128 slices)
inter_rows = reg["intermediate"][: rows_cap * wpr * 4].view(rows_cap, wpr * 4)[:P].cpu().numpy()     # bytes
inter_codes = np.concatenate([inter_rows[:, s * 128:(s + 1) * 128] for s in range(4)] + [inter_rows[:, 512:576]], axis=1)
pad_tail = inter_rows[:, 576:640]
isc = reg["intermediate"][rows_cap * wpr * 4: rows_cap * wpr * 4 + 5 * rows_cap * 4].view(5, rows_cap, 4)[:, :P].cpu().numpy()
inter_scales = np.concatenate([isc[s] for s in range(5)], axis=1)[:, :18]   # [P, 20] -> 18 real groups
pad_scales = np.concatenate([isc[s] for s in range(5)], axis=1)[:, 18:]
route = reg["route_output"].view(torch.bfloat16).view(-1, K)[:P].view(torch.int16).cpu().numpy()
out_bits = out.view(torch.int16).cpu().numpy()
scal = {k: got[k].float().cpu().numpy() for k in ("alpha1", "input_scale", "alpha2", "down_scale")}
print("alpha1 unique", np.unique(scal["alpha1"]), "input_scale", np.unique(scal["input_scale"]),
      "alpha2", np.unique(scal["alpha2"]), "down_scale", np.unique(scal["down_scale"]))
print(f"intermediate padding: tail codes nonzero {int((pad_tail != 0).sum())}, tail scale bytes nonzero {int((pad_scales != 0).sum())}")

# TopKSum: the output vs the fp32 sum of the pairs' rows (informational)
route_f = torch.from_numpy(route.astype(np.int32) << 16).view(torch.float32).view(T, TOPK, K)
out_f = torch.from_numpy(out_bits.astype(np.int32) << 16).view(torch.float32)
s_seq = route_f[:, 0].clone()
for k in range(1, TOPK):
    s_seq = s_seq + route_f[:, k]
seq_bf16 = s_seq.to(torch.bfloat16).float()
print(f"TopKSum vs sequential fp32 sum of the rows -> bf16: {int((seq_bf16 != out_f).sum())} of {out_f.numel()} differ")

# graph replay stability
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        run()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    run()
same = 0
for _ in range(3):
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    same += int(np.array_equal(out.view(torch.int16).cpu().numpy(), out_bits))
route2 = reg["route_output"].view(torch.bfloat16).view(-1, K)[:P].view(torch.int16).cpu().numpy()
print(f"CUDA graph replays: output identical {same}/3; route_output after the replays identical {np.array_equal(route2, route)}")

# layout sanity: which half of the input w13 feeds the gate half of the projections (projections[:, :N] is what
# b12x's activation applies the sigmoid to). Measured: the FIRST half (so vLLM's [w1; w3] stacking is the gate first)
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def deq(w, sc):
    w = torch.from_numpy(w.astype(np.int64))
    v = torch.stack((FP4[w & 0xF], FP4[w >> 4]), dim=-1).flatten(-2)
    return v * torch.exp2(torch.from_numpy(sc.astype(np.float32)) - 127.0).repeat_interleave(32, dim=-1)


xa = torch.from_numpy(a_values.copy()).view(torch.float8_e4m3fn).float()
xa = xa * torch.exp2(torch.from_numpy(a_scales.astype(np.float32)) - 127.0).repeat_interleave(32, dim=1)
proj_f = torch.from_numpy(proj.astype(np.int32) << 16).view(torch.float32).view(T, TOPK, 2 * N)
ids_np = Z["ids"]
cos_gate_w1 = cos_gate_w3 = 0.0
for t in range(T):
    e = int(ids_np[t, 0])
    first_ref = deq(Z["w13"][e][:N], Z["w13s"][e][:N]) @ xa[t]
    second_ref = deq(Z["w13"][e][N:], Z["w13s"][e][N:]) @ xa[t]
    gp = proj_f[t, 0, :N]
    cos_gate_w1 += torch.nn.functional.cosine_similarity(gp, first_ref, dim=0).item() / T
    cos_gate_w3 += torch.nn.functional.cosine_similarity(gp, second_ref, dim=0).item() / T
print(f"projections[:, :N] (the gate half) vs fp32 math with the first half of w13: cosine {cos_gate_w1:.5f}; "
      f"with the second half: {cos_gate_w3:.5f}")

np.savez(outp, a_values=a_values, a_scales=a_scales, projections=proj, inter_codes=inter_codes, inter_scales=inter_scales,
         route_output=route, out=out_bits, ids=Z["ids"], weights=Z["weights"], **{f"scal_{k}": v for k, v in scal.items()},
         graph_same=same, cos_gate_first=cos_gate_w1, cos_gate_second=cos_gate_w3, impl=str(plan.launch_plan.implementation))
print(f"wrote {outp}")
