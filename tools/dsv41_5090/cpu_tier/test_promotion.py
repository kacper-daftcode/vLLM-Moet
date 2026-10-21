"""Promote-on-miss end to end (phase 2.2): the resident pools in b12x's prepared layout, the repack of a record into
a slot (bit-exact against prepare_weights), the LRU placement manager's promotions on a copy stream between captured
steps, and the ordering protocol checked step by step against the tier's miss / use logs.

In the serving image on one sm_120 GPU under its lock (see test_cpu_tier.py for the docker line); e.g.
    ... test_promotion.py --records /parity/b12x_inputs_l1.npz --tables /parity --steps 240 --cpus 8-15 --dispatcher-cpu 40
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "cpu_moe"))


def parse_cpus(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def ensure_libs(args):
    if os.environ.get("CPU_TIER_LIB"):
        return
    out = os.path.join(args.build_dir, "libcpu_tier.so")
    if not os.path.exists(out) or args.rebuild:
        subprocess.run(["bash", os.path.join(_HERE, "build.sh")], check=True, env=dict(os.environ, ARCH=args.arch, OUT=out))
    os.environ["CPU_TIER_LIB"] = out
    os.environ["DSV41_B12X_LIB"] = out


def dequant_fp4(nib: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    fp4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=nib.device)
    idx = torch.stack((nib & 0xF, nib >> 4), dim=-1).flatten(-2).long()
    return fp4[idx] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)


class StandIn:
    def __init__(self, split: dict, dev):
        E = split["w1n"].shape[0]
        self.w1 = torch.stack([dequant_fp4(split["w1n"][e].to(dev), split["w1s"][e].to(dev)) for e in range(E)])
        self.w3 = torch.stack([dequant_fp4(split["w3n"][e].to(dev), split["w3s"][e].to(dev)) for e in range(E)])
        self.w2 = torch.stack([dequant_fp4(split["w2n"][e].to(dev), split["w2s"][e].to(dev)) for e in range(E)])

    def __call__(self, x_f32, ids, remapped, wts, route):
        T, K = ids.shape
        e = ids.reshape(-1).clamp(min=0).long()
        xt = x_f32.repeat_interleave(K, dim=0)
        g = torch.einsum("pk,pnk->pn", xt, self.w1[e]).clamp(max=10.0)
        u = torch.einsum("pk,pnk->pn", xt, self.w3[e]).clamp(-10.0, 10.0)
        o = torch.einsum("pn,pkn->pk", torch.nn.functional.silu(g) * u, self.w2[e]) * wts.reshape(-1, 1)
        keep = (remapped.reshape(-1) >= 0).to(o.dtype).unsqueeze(1)
        route.copy_((o * keep).to(torch.bfloat16))


def reference_prepared(Z, dev) -> dict:
    """b12x's prepared tensors for all E experts of the input file, per expert, through the compact-path spy."""
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe import _impl, _preparation
    import b12x.moe._shared.kernels.w4a8_compact_micro as cm

    E, N2, HALF = Z["w13"].shape
    N, K = N2 // 2, HALF * 2
    w13, w13s, w2, w2s = (torch.from_numpy(np.ascontiguousarray(Z[k])).to(dev) for k in ("w13", "w13s", "w2", "w2s"))
    plan_w = fused_moe.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                    params_dtype=torch.bfloat16, num_experts=E, hidden_size=K, intermediate_size=N, w13_layout="w31")
    one = torch.ones(E, dtype=torch.float32, device=dev)
    prepared = fused_moe.prepare_weights(plan=plan_w, w1_fp4=w13, w1_blockscale=w13s, w1_global_scale=one, a1_gscale=one,
                                         w2_fp4=w2, w2_blockscale=w2s, w2_global_scale=one, a2_gscale=one, params_dtype=torch.bfloat16)
    caps = fused_moe.Caps(max_tokens=1, num_topk=6, device=dev, weight_plan=prepared.plan, core_token_counts=(1,),
                          route_num_experts=0, quant_mode="w4a8_mx", apply_router_weight_on_input=False, swiglu_limit=10.0,
                          swiglu_alpha=None, swiglu_beta=None, frozen=True)
    plan = fused_moe.plan(caps)
    orig_bind = _impl.TPMoEScratchPlan.bind

    def bind_with_compact(self, **kw):
        b = orig_bind(self, **kw)
        c = _preparation._compact_launches(self.launch_plan, self.caps)
        return dataclasses.replace(b, compact_launches=c) if c is not None else b

    _impl.TPMoEScratchPlan.bind = bind_with_compact
    got = {}
    orig = cm.launch_w4a8_compact_micro

    def spy(**kw):
        got.update(kw)
        return orig(**kw)

    cm.launch_w4a8_compact_micro = spy
    (spec,) = plan.scratch_specs()
    scratch = torch.zeros(int(spec.shape[0]), dtype=torch.uint8, device=dev)
    out = torch.empty(1, K, dtype=torch.bfloat16, device=dev)
    b = fused_moe.bind(plan, scratch=scratch, a=torch.zeros(1, K, dtype=torch.bfloat16, device=dev), experts=prepared,
                       topk_weights=torch.ones(1, 6, device=dev), topk_ids=torch.zeros(1, 6, dtype=torch.int32, device=dev),
                       output=out, input_scales_static=True, unit_scale_contract=False)
    fused_moe.run(binding=b)
    torch.cuda.synchronize()
    cm.launch_w4a8_compact_micro = orig
    _impl.TPMoEScratchPlan.bind = orig_bind
    ref = {name: got[key].contiguous().view(E, -1).clone() for name, key in
           (("w13", "w13"), ("w13s", "w13_scales"), ("w2", "w2"), ("w2s", "w2_scales"))}
    del prepared, scratch, plan
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--resident", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--cpus", default="8-15")
    ap.add_argument("--dispatcher-cpu", type=int, default=40)
    ap.add_argument("--build-dir", default="/tmp")
    ap.add_argument("--arch", default="znver4")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ensure_libs(args)
    os.environ["DSV41_B12X_TABLES"] = args.tables
    import dsv41_b12x as B
    from cpu_tier import CpuTier
    from placement_manager import PlacementManager, ResidentPool, load_layout

    dev = torch.device("cuda", 0)
    Z = np.load(args.records)
    records = Z["records"]
    E, R, L, T, TOPK = records.shape[0], args.resident, args.layers, args.tokens, 6
    store = torch.from_numpy(records).pin_memory()
    split = {k: torch.stack([torch.from_numpy(np.ascontiguousarray(B.split_record(records[e])[k])) for e in range(E)])
             for k in ("w1n", "w1s", "w3n", "w3s", "w2n", "w2s")}
    stand_in = StandIn(split, dev)
    layout = load_layout(args.tables, dev)
    ref = reference_prepared(Z, dev)
    print(f"store {E} records; pools R={R} x {L} layers; reference prepared blocks {[tuple(v.shape) for v in ref.values()]}", flush=True)

    tier = CpuTier(L, E, parse_cpus(args.cpus), args.dispatcher_cpu, dev, args.tables, timeout_ms=500.0)
    for layer in range(L):
        for e in range(E):
            tier.set_record(layer, e, store.data_ptr() + e * B.RECORD_BYTES)
    pools = [ResidentPool(layer, E, R, dev) for layer in range(L)]
    copy_stream = torch.cuda.Stream()
    mgr = PlacementManager(tier.ext, layout, pools, lambda layer, e: store[e], copy_stream, max_promotions_per_layer=3)
    for layer in range(L):
        mgr.fill_initial(layer, [(e + 4 * layer) % E for e in range(R)])

    # ---- 1. the repack: every slot of the initial placement == prepare_weights' block for its expert ------------
    def pool_matches(layer) -> tuple[int, int]:
        P = pools[layer]
        bad = n = 0
        for s, e in enumerate(P.expert_of_slot):
            if e < 0:
                continue
            for name in ("w13", "w13s", "w2", "w2s"):
                n += 1
                bad += int(not torch.equal(P.pool[name][s], ref[name][e]))
        return bad, n

    tot_bad = tot_n = 0
    for layer in range(L):
        b_, n_ = pool_matches(layer)
        tot_bad += b_
        tot_n += n_
    # the slot table on the device == the host mirror
    tab_ok = all(torch.equal(pools[l].slot_table.cpu(), torch.tensor([pools[l].mapped(e) for e in range(E)], dtype=torch.int32)) for l in range(L))
    print(f"1. initial placement: {tot_bad} of {tot_n} prepared blocks differ from prepare_weights; slot tables consistent {tab_ok}", flush=True)
    ok1 = tot_bad == 0 and tab_ok
    # promotion timing (copy stream, H2D 4.7 MB + repack), 20 experts in a spare slot-like staging run
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(copy_stream):
        ev0.record()
        for i in range(20):
            mgr._repack_into(pools[0], 0, i % E, 0)
        ev1.record()
    copy_stream.synchronize()
    t_promo = ev0.elapsed_time(ev1) / 20
    mgr.fill_initial(0, [e for e in range(R)])  # restore slot 0's expert
    print(f"   promotion (pinned H2D + repack): {t_promo * 1e3:.0f} us per expert", flush=True)

    # ---- 2. captured steps with promotions between them ----------------------------------------------------------
    tier.start(first_step=1)
    x_bf16 = torch.zeros(T, B.HID, dtype=torch.bfloat16, device=dev)
    a_values = torch.zeros(8, B.HID, dtype=torch.uint8, device=dev)
    a_scales = torch.zeros(8, B.B1, dtype=torch.uint8, device=dev)
    ids = torch.zeros(T, TOPK, dtype=torch.int32, device=dev)
    wts = torch.zeros(T, TOPK, dtype=torch.float32, device=dev)
    remapped = [torch.zeros(T, TOPK, dtype=torch.int32, device=dev) for _ in range(L)]
    route = [torch.zeros(T * TOPK, B.HID, dtype=torch.bfloat16, device=dev) for _ in range(L)]
    g = torch.Generator().manual_seed(9)
    favored = [0]

    def routing(step: int):
        if step % 40 == 0:
            favored[0] = torch.randperm(E, generator=g)[:6].tolist()
        out = np.zeros((T, TOPK), dtype=np.int32)
        for t in range(T):
            pool_ = list(favored[0]) if torch.rand(1, generator=g).item() < 0.7 else list(range(E))
            perm = torch.randperm(len(pool_), generator=g).tolist()
            picks = [pool_[i] for i in perm][:TOPK]
            while len(picks) < TOPK:
                c = int(torch.randint(0, E, (1,), generator=g))
                if c not in picks:
                    picks.append(c)
            out[t] = picks
        return out

    def new_inputs(step: int):
        x = torch.randn(T, B.HID, generator=g)
        heavy = torch.randperm(B.HID, generator=g)[:48]
        x[:, heavy] *= 12.0
        xb = x.to(torch.bfloat16)
        q, s = B.rows_quant(xb, zero_byte=0)
        i = routing(step)
        w = torch.rand(T, TOPK, generator=g) * 0.9 + 0.1
        x_bf16.copy_(xb)
        a_values[:T].copy_(torch.from_numpy(q))
        a_scales[:T].copy_(torch.from_numpy(s))
        ids.copy_(torch.from_numpy(i))
        wts.copy_(w)
        return q, s, i, w.numpy()

    def step():
        tier.step_begin()
        xf = x_bf16.float()
        for layer in range(L):
            tier.publish(layer, ids, wts, pools[layer].slot_table, a_values, a_scales, T, remapped[layer])
            stand_in(xf, ids, remapped[layer], wts, route[layer])
            tier.wait_inject(layer, route[layer])

    sgraph = torch.cuda.Stream()
    sgraph.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sgraph):
        new_inputs(0)
        step()
    torch.cuda.current_stream().wait_stream(sgraph)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()
    tier.misses()
    tier.uses()

    # host model at every launch: per layer the experts surely resident (mapped, map event already waited), and the
    # pending ones (promoted after the previous step: visible to this step or not, depending on the copy stream)
    resident_at_launch = {}
    pending_at_launch = {}
    routing_at = {}
    bad_rows = checked_rows = 0
    first_step = int(tier.step_dev.item()) + 1
    for it in range(args.steps):
        step_no = first_step + it
        mgr.before_launch()
        pend = [set(mgr.last_promoted.get(l, set())) for l in range(L)]
        resident_at_launch[step_no] = [set(pools[l].slot_of_expert.keys()) - pend[l] for l in range(L)]
        pending_at_launch[step_no] = pend
        q, s, i_np, w_np = new_inputs(it)
        routing_at[step_no] = i_np
        graph.replay()
        if it % 60 == 7:  # the injected rows against the reference on a few steps (surely-missing pairs only)
            torch.cuda.synchronize()
            for layer in range(L):
                rows = route[layer].view(torch.int16).cpu().numpy().view(np.uint16)
                resident = resident_at_launch[step_no][layer] | pend[layer]
                for t in range(T):
                    for k in range(TOPK):
                        e = int(i_np[t, k])
                        if e in resident:
                            continue
                        r = B.reference_pair(records[e], q[t], s[t], float(w_np[t, k]))
                        checked_rows += 1
                        bad_rows += int(not np.array_equal(rows[t * TOPK + k], r["out"]))
        mgr.note_uses(tier.uses())
        mgr.after_launch(tier.misses(), step_no)
    torch.cuda.synchronize()
    time.sleep(0.05)
    tier.uses()
    # ---- 3. the ordering protocol: every served miss was unmapped at its launch, every unmapped pick was served -----
    all_misses = {}
    tier._miss_cursor.value = 0
    for row in tier.misses(cap=65536):
        all_misses.setdefault((int(row[0]), int(row[1])), set()).add(int(row[2]))
    proto_bad = 0
    hits = picks = ambiguous = 0
    for step_no, res in resident_at_launch.items():
        i_np = routing_at[step_no]
        pend = pending_at_launch[step_no]
        for layer in range(L):
            picked = set(int(e) for e in i_np.reshape(-1))
            served = all_misses.get((step_no, layer), set())
            surely_missing = {e for e in picked if e not in res[layer] and e not in pend[layer]}
            surely_resident = {e for e in picked if e in res[layer]}
            maybe = {e for e in picked if e in pend[layer]}
            viol = (not surely_missing <= served) or (served & surely_resident) or (not served <= surely_missing | maybe)
            if viol:
                proto_bad += 1
                if proto_bad <= 3:
                    print(f"   step {step_no} layer {layer}: served {sorted(served)} surely missing {sorted(surely_missing)} "
                          f"surely resident {sorted(surely_resident)} pending {sorted(maybe)}", flush=True)
            ambiguous += len(maybe)
            picks += i_np.size
            hits += sum(1 for e in i_np.reshape(-1) if int(e) in res[layer])
    # pools after the run: bytes and tables
    tot_bad = tot_n = 0
    for layer in range(L):
        b_, n_ = pool_matches(layer)
        tot_bad += b_
        tot_n += n_
    tab_ok = all(torch.equal(pools[l].slot_table.cpu(), torch.tensor([pools[l].mapped(e) for e in range(E)], dtype=torch.int32)) for l in range(L))
    tot = tier.totals()
    print(f"2. {args.steps} captured steps with promotions: {mgr.stats}; ordering protocol violations {proto_bad} of "
          f"{len(resident_at_launch) * L} (step, layer) ({ambiguous} picks of experts whose map was still in flight); "
          f"hit rate against the surely-resident set {hits / max(picks, 1):.3f} (a static {R}/{E} placement: {R / E:.3f} on "
          f"uniform routing); injected rows checked {checked_rows}, wrong {bad_rows}; tier errors {tot['errors']}, "
          f"error word {tier.error()}", flush=True)
    print(f"3. pools after the run: {tot_bad} of {tot_n} prepared blocks differ from prepare_weights; slot tables consistent {tab_ok}", flush=True)
    ok = ok1 and proto_bad == 0 and bad_rows == 0 and tot_bad == 0 and tab_ok and tot["errors"] == 0 and tier.error() == 0
    tier.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
