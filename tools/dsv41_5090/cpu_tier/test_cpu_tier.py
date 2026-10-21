"""The CPU expert tier end to end on one GPU, without vLLM: publish -> (GPU stand-in for the resident experts) ->
wait + inject, eagerly and from one CUDA graph per step, against the b12x-exact reference; timings; fail-closed.

In the serving image (torch 2.14 + CUDA 13, nvcc for the extension), one sm_120 GPU under its lock, e.g.
    flock -w 3600 /tmp/exl3diff-gpu1.lock docker run --rm --gpus '"device=1"' --name moet-tier-test \
      -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache -v /mnt/nvme/vLLM-Moet-cpu:/repo -v <parity dir>:/parity \
      --entrypoint python3 <image> /repo/tools/dsv41_5090/cpu_tier/test_cpu_tier.py --records /parity/b12x_inputs_l1.npz \
      --tables /parity --layers 4 --steps 300 --cpus 8-15 --dispatcher-cpu 40

What it checks:
  1. eager steps: every missing pair's route_output row == the scalar reference (bit for bit); resident rows untouched
  2. a captured step (step counter + L x (publish, stand-in, wait+inject)) replayed with fresh routings and tokens:
     the same identity on sampled steps; per-step GPU time with and without the tier's ops; host KPIs (busy time per
     layer, pairs, experts)
  3. fail-closed: the dispatcher stopped -> one replay arms the error word within the timeout and completes; the next
     replay completes at once; restart -> correct again
"""
from __future__ import annotations

import argparse
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
    """Build libcpu_tier.so for this machine's compiler if CPU_TIER_LIB is not set (the host build may not load here)."""
    if os.environ.get("CPU_TIER_LIB"):
        return
    out = os.path.join(args.build_dir, "libcpu_tier.so")
    if not os.path.exists(out) or args.rebuild:
        env = dict(os.environ, ARCH=args.arch, OUT=out)
        subprocess.run(["bash", os.path.join(_HERE, "build.sh")], check=True, env=env)
    os.environ["CPU_TIER_LIB"] = out
    os.environ["DSV41_B12X_LIB"] = out


def dequant_fp4(nib: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """[R, K/2] uint8 + [R, K/32] uint8 -> fp32 [R, K] on the tensors' device."""
    fp4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=nib.device)
    idx = torch.stack((nib & 0xF, nib >> 4), dim=-1).flatten(-2).long()
    return fp4[idx] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)


class StandIn:
    """The GPU's resident experts, approximated: fp32 math on dequantized weights (the handoff is what is under test;
    the bits of the resident rows are the stand-in's own and only checked for being left alone)."""

    def __init__(self, records_split: dict, dev):
        E = records_split["w1n"].shape[0]
        self.w1 = torch.stack([dequant_fp4(records_split["w1n"][e].to(dev), records_split["w1s"][e].to(dev)) for e in range(E)])
        self.w3 = torch.stack([dequant_fp4(records_split["w3n"][e].to(dev), records_split["w3s"][e].to(dev)) for e in range(E)])
        self.w2 = torch.stack([dequant_fp4(records_split["w2n"][e].to(dev), records_split["w2s"][e].to(dev)) for e in range(E)])

    def __call__(self, x_f32: torch.Tensor, ids: torch.Tensor, remapped: torch.Tensor, wts: torch.Tensor, route: torch.Tensor):
        T, K = ids.shape
        e = ids.reshape(-1).clamp(min=0).long()
        xt = x_f32.repeat_interleave(K, dim=0)                     # [pairs, hid]
        g = torch.einsum("pk,pnk->pn", xt, self.w1[e]).clamp(max=10.0)
        u = torch.einsum("pk,pnk->pn", xt, self.w3[e]).clamp(-10.0, 10.0)
        h = torch.nn.functional.silu(g) * u
        o = torch.einsum("pn,pkn->pk", h, self.w2[e]) * wts.reshape(-1, 1)
        keep = (remapped.reshape(-1) >= 0).to(o.dtype).unsqueeze(1)  # missing pairs -> zero rows, as b12x
        route.copy_((o * keep).to(torch.bfloat16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="b12x_inputs*.npz (its `records` array = the expert store)")
    ap.add_argument("--tables", required=True, help="dir with b12x_act_m_limit10.f32 / b12x_ue8m0_inv.f32")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--cpus", default="8-15")
    ap.add_argument("--dispatcher-cpu", type=int, default=40)
    ap.add_argument("--timeout-ms", type=float, default=500.0)
    ap.add_argument("--build-dir", default="/tmp")
    ap.add_argument("--arch", default="znver4")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ensure_libs(args)
    os.environ["DSV41_B12X_TABLES"] = args.tables
    import dsv41_b12x as B  # noqa: E402
    from cpu_tier import CpuTier  # noqa: E402

    dev = torch.device("cuda", 0)
    Z = np.load(args.records)
    records = Z["records"]
    E = records.shape[0]
    T, TOPK, L = args.tokens, 6, args.layers
    assert T <= 8
    store = torch.from_numpy(records).pin_memory()  # the pinned expert store [E, RECORD_BYTES]
    split = {k: torch.stack([torch.from_numpy(np.ascontiguousarray(v)) for v in
                             (B.split_record(records[e])[k] for e in range(E))]) for k in ("w1n", "w1s", "w3n", "w3s", "w2n", "w2s")}
    stand_in = StandIn(split, dev)
    print(f"store: {E} records ({E * B.RECORD_BYTES / 2**20:.0f} MiB pinned); layers {L}, tokens {T}, top-{TOPK}", flush=True)

    tier = CpuTier(L, E, parse_cpus(args.cpus), args.dispatcher_cpu, dev, args.tables, timeout_ms=args.timeout_ms)
    for layer in range(L):
        for e in range(E):
            tier.set_record(layer, e, store.data_ptr() + e * B.RECORD_BYTES)
    # resident subsets: layer l keeps the experts with (e + l) % 3 != 0 (about two thirds)
    slot_tables = []
    for layer in range(L):
        slot = torch.full((E,), -1, dtype=torch.int32)
        k = 0
        for e in range(E):
            if (e + layer) % 3 != 0:
                slot[e] = k
                k += 1
        slot_tables.append(slot.to(dev))
    tier.start(first_step=1)

    # static tensors of the step
    x_bf16 = torch.zeros(T, B.HID, dtype=torch.bfloat16, device=dev)
    a_values = torch.zeros(16, B.HID, dtype=torch.uint8, device=dev)
    a_scales = torch.zeros(16, B.B1, dtype=torch.uint8, device=dev)
    ids = torch.zeros(T, TOPK, dtype=torch.int32, device=dev)
    wts = torch.zeros(T, TOPK, dtype=torch.float32, device=dev)
    remapped = [torch.zeros(T, TOPK, dtype=torch.int32, device=dev) for _ in range(L)]
    route = [torch.zeros(T * TOPK, B.HID, dtype=torch.bfloat16, device=dev) for _ in range(L)]
    g = torch.Generator().manual_seed(5)

    def new_inputs():
        x = torch.randn(T, B.HID, generator=g)
        heavy = torch.randperm(B.HID, generator=g)[:48]
        x[:, heavy] *= 12.0
        xb = x.to(torch.bfloat16)
        q, s = B.rows_quant(xb, zero_byte=0)
        i = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32)
        w = torch.rand(T, TOPK, generator=g) * 0.9 + 0.1
        x_bf16.copy_(xb)
        a_values[:T].copy_(torch.from_numpy(q))
        a_scales[:T].copy_(torch.from_numpy(s))
        ids.copy_(i)
        wts.copy_(w)
        return q, s, i.numpy(), w.numpy()

    def step(with_tier=True):
        if with_tier:
            tier.step_begin()
        xf = x_bf16.float()
        for layer in range(L):
            if with_tier:
                tier.publish(layer, ids, wts, slot_tables[layer], a_values, a_scales, T, remapped[layer])
            else:
                remapped[layer].copy_(torch.where(ids >= 0, slot_tables[layer][ids.long()], -1))
            stand_in(xf, ids, remapped[layer], wts, route[layer])
            if with_tier:
                tier.wait_inject(layer, route[layer])

    def verify(q, s, i_np, w_np, label):
        torch.cuda.synchronize()
        bad = 0
        n_missing = 0
        for layer in range(L):
            slot = slot_tables[layer].cpu().numpy()
            rows = route[layer].view(torch.int16).cpu().numpy().view(np.uint16)
            # the stand-in's own rows for the resident pairs (recomputed here for the comparison)
            keep_rows = route[layer].clone()
            stand_in(x_bf16.float(), ids, remapped[layer], wts, keep_rows)
            keep = keep_rows.view(torch.int16).cpu().numpy().view(np.uint16)
            for t in range(T):
                for k in range(TOPK):
                    p = t * TOPK + k
                    e = int(i_np[t, k])
                    if slot[e] >= 0:
                        bad += int(not np.array_equal(rows[p], keep[p]))
                    else:
                        n_missing += 1
                        ref = B.reference_pair(records[e], q[t], s[t], float(w_np[t, k]))
                        d = int((rows[p] != ref["out"]).sum())
                        bad += d > 0
                        if d and bad <= 3:
                            print(f"    {label}: layer {layer} pair {p} (expert {e}) differs in {d} columns", flush=True)
        return bad, n_missing

    # ---- 1. eager -----------------------------------------------------------------------------------------------
    total_bad = total_missing = 0
    for it in range(3):
        q, s, i_np, w_np = new_inputs()
        step()
        bad, nm = verify(q, s, i_np, w_np, f"eager {it}")
        total_bad += bad
        total_missing += nm
    err = tier.error()
    print(f"1. eager: {total_missing} missing pairs over 3 steps x {L} layers, rows differing from the reference: "
          f"{total_bad}; resident rows disturbed: counted in the same number; error word {err}", flush=True)
    ok1 = total_bad == 0 and err == 0

    # ---- 2. captured ----------------------------------------------------------------------------------------------
    sgraph = torch.cuda.Stream()
    sgraph.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sgraph):
        new_inputs()
        step()  # warm-up on the side stream (the extension's first launches)
    torch.cuda.current_stream().wait_stream(sgraph)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    new_inputs()
    with torch.cuda.graph(graph):
        step()
    graph_plain = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_plain):
        step(with_tier=False)
    torch.cuda.synchronize()
    ok2 = True
    times, times_plain = [], []
    checked = bad2 = missing2 = 0
    t_begin = tier.now_ns()
    for it in range(args.steps):
        q, s, i_np, w_np = new_inputs()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        if (it + 1) % args.check_every == 0 or it < 2:
            bad, nm = verify(q, s, i_np, w_np, f"replay {it}")
            checked += 1
            bad2 += bad
            missing2 += nm
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
        if tier.error():
            print(f"    error word {tier.error()} after replay {it}", flush=True)
            ok2 = False
            break
    torch.cuda.synchronize()
    wall_ms = (tier.now_ns() - t_begin) / 1e6
    for it in range(50):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph_plain.replay()
        e1.record()
        e1.synchronize()
        times_plain.append(e0.elapsed_time(e1))
    tot = tier.totals()
    kp = tier.kpis()
    busy = (kp[:, 6] - kp[:, 5]) / 1e3  # us per (step, layer)
    per_layer_pairs = kp[:, 2]
    print(f"2. captured step replayed {len(times)} x: GPU time per step median {np.median(times):.3f} ms (plain stand-in "
          f"{np.median(times_plain):.3f} ms -> tier overhead {np.median(times) - np.median(times_plain):.3f} ms per step for "
          f"{L} layers incl. the CPU time on the critical path); wall {wall_ms / max(1, len(times)):.2f} ms per step "
          f"(host-side input generation included); checked {checked} steps: {missing2} missing pairs, {bad2} rows wrong; "
          f"tier totals {tot}; host busy per layer: median {np.median(busy):.0f} us, p90 {np.percentile(busy, 90):.0f} us, "
          f"max {busy.max():.0f} us at pairs median {np.median(per_layer_pairs):.0f} max {per_layer_pairs.max()}", flush=True)
    ok2 = ok2 and bad2 == 0 and tier.error() == 0

    # ---- 3. fail-closed -----------------------------------------------------------------------------------------
    tier.stop()
    new_inputs()
    t0 = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0
    err1 = tier.error()
    t0 = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    t_second = time.perf_counter() - t0
    tot2 = tier.totals()
    # restart at the next step the graph will publish (the counter is already past the two failed steps)
    tier.clear_error()
    tier.start(first_step=int(tier.step_dev.item()) + 1)
    q, s, i_np, w_np = new_inputs()
    graph.replay()
    bad3, nm3 = verify(q, s, i_np, w_np, "after restart")
    print(f"3. fail-closed: dispatcher stopped -> replay took {t_first * 1e3:.0f} ms (timeout {args.timeout_ms:.0f} ms), error "
          f"word {err1}; second replay {t_second * 1e3:.1f} ms (short-circuit); restarted at step {tot2['expected_step']}: "
          f"{nm3} missing pairs, {bad3} wrong, error word {tier.error()}", flush=True)
    ok3 = err1 != 0 and t_first < args.timeout_ms / 1e3 * L + 1.0 and t_second < 0.05 and bad3 == 0 and tier.error() == 0
    tier.close()
    passed = ok1 and ok2 and ok3
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
