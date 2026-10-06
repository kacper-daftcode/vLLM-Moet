#!/usr/bin/env python3
"""NCCL serialEvent records reduced in captured graphs (nccl_graph_records.py, patch_vllm_nccl_graph_records.py).

op part (4 ranks; the patch is not needed): a decode-like body - per layer a GEMV v3 [6, 5120 <- 2048], the join of an
mHC-like side stream, vLLM's all-reduce, an add, two side-stream kernels; an all-gather every 10 layers - captured
twice, as captured and with all but the last NCCL record node dropped (keep_graph=True, then instantiate). The two
graphs must give the same bits on wide-range random inputs; the rewritten one must keep one record node per event,
its first event-wait node and the NCCL chain (every NCCL kernel after the first has an NCCL kernel parent); the
replay time of both is printed. Then the rewritten graphs (a decode-like one: 40 all-reduces + 2 all-gathers, a
drafter-like one: 7 all-reduces) are replayed interleaved with eager collectives on the same stream without host
synchronisation, on small-integer inputs whose sums are exact: every output must be exact.

integration part (when the installed vllm/compilation/cuda_graph.py is patched): the real CUDAGraphWrapper in FULL
mode around a module running the body, captured and replayed through vLLM's forward context: replays bit-identical
to the eager forward; with the patch the captured graph keeps one record node per event, with
VLLM_MOET_NCCL_GRAPH_RECORDS=0 it is captured as before (keep_graph off).

On one GPU (no NCCL kernels in the graphs) both parts run with nothing to drop - the capture path is still checked.

usage (serving image): python3 patch_vllm_nccl_graph_records.py && torchrun --nproc_per_node=4 test_nccl_graph_records_tp4.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _d in (HERE.parent / "sm120_gemv", Path("/opt/vllm-moet/dsv41_sm120/sm120_gemv")):
    if (_d / "mxfp8_gemv_sm120.py").exists():
        sys.path.insert(0, str(_d))
        break

T, H, LAYERS = 6, 5120, 40


def main() -> int:
    global LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=LAYERS, help="layers of the op-part graph (80 = a decode step)")
    LAYERS = ap.parse_args().layers
    for key, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29541"), ("RANK", "0"), ("WORLD_SIZE", "1"),
                       ("LOCAL_RANK", "0")):
        os.environ.setdefault(key, value)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", torch.cuda.current_device())

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils.torch_utils import set_default_torch_dtype

    from graph_topology import Topology
    from mxfp8_gemv_sm120 import mxfp8_gemv
    from nccl_gap_tp4 import mx_weight
    from nccl_graph_records import drop_intermediate_nccl_records

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    vllm_config = VllmConfig()
    ok = True
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        cpu_group = get_tp_group().cpu_group

        def all_ok(flag: bool) -> bool:
            t = torch.tensor([0 if flag else 1])
            if world > 1:
                dist.all_reduce(t, group=cpu_group)
            return t.item() == 0

        def barrier() -> None:
            if world > 1:
                dist.barrier(group=cpu_group)

        log(f"{torch.cuda.get_device_name(dev)} x {world}")
        g = torch.Generator(device=dev).manual_seed(3)
        ws = [mx_weight(H, 2048, dev, g) for _ in range(LAYERS)]
        a, a_sf = mxfp8_e4m3_quantize(torch.randn(T, 2048, device=dev, generator=g).to(torch.bfloat16),
                                      is_sf_swizzled_layout=True)
        resid = torch.zeros(T, H, dtype=torch.bfloat16)
        side_buf = torch.zeros(T, H, dtype=torch.bfloat16)
        side = torch.cuda.Stream()

        def layers(lo: int, hi: int) -> None:
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            for i in range(lo, hi):
                y = mxfp8_gemv(a, a_sf, *ws[i])
                cur.wait_stream(side)  # the side stream joins on the all-reduce, as with fuse_mhc_all_reduce
                resid.add_(tensor_model_parallel_all_reduce(y))
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    side_buf.mul_(0.5).add_(resid)
                    side_buf.mul_(0.25)
                if i % 10 == 9:
                    gathered = tensor_model_parallel_all_gather(resid[:, :1280].contiguous(), dim=-1)
                    resid[:, : 1280 * world].add_(gathered * 0.25)
            cur.wait_stream(side)

        def body() -> tuple[torch.Tensor, torch.Tensor]:
            layers(0, LAYERS)
            return resid.clone(), side_buf.clone()

        def wide(shape) -> torch.Tensor:
            x = torch.randn(shape, device=dev, generator=g)
            return (x * torch.exp2(torch.randint(-6, 7, shape, device=dev, generator=g).float())).to(torch.bfloat16)

        def capture(fn, rewrite: bool):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=rewrite)
                with torch.cuda.graph(graph, stream=stream):
                    out = fn()
            dropped = found = 0
            if rewrite:
                topo_before = Topology(graph.raw_cuda_graph())
                t0 = time.perf_counter()
                dropped, found = drop_intermediate_nccl_records(graph.raw_cuda_graph())
                log(f"  rewrite of a {len(topo_before.nodes)}-node graph: {(time.perf_counter() - t0) * 1e3:.1f} ms")
                topo = Topology(graph.raw_cuda_graph())
                graph.instantiate()
                return graph, out, (topo_before, topo, dropped, found)
            return graph, out, None

        # ---- op part: as captured vs rewritten, same bits; topology; timing
        g_plain, out_plain, _ = capture(body, rewrite=False)
        g_new, out_new, (tb, tn, dropped, found) = capture(body, rewrite=True)
        n_nccl = len(tn.nccl_nodes())
        n_rec = sum(1 for nm in tn.name if nm == "event-record")
        n_wait = sum(1 for nm in tn.name if nm == "event-wait")
        chain = all(any(tn.is_nccl(p) for p in tn.parents[i]) for i in tn.nccl_nodes()[1:])
        topo_ok = (n_nccl == 0 and found == 0) or (n_rec == 1 and n_wait == sum(1 for nm in tb.name if nm == "event-wait")
                                                  and dropped == found - 1 and chain)
        log(f"[op] {n_nccl} NCCL kernels, {found} record nodes as captured -> {dropped} dropped, {n_rec} record / "
            f"{n_wait} wait nodes left, NCCL chain kept: {chain} -> {'OK' if topo_ok else 'UNEXPECTED'}")
        ok &= all_ok(topo_ok)
        same = True
        torch.cuda.synchronize()
        for _ in range(4):
            r0, s0 = wide((T, H)), wide((T, H))
            outs = []
            for graph, out in ((g_plain, out_plain), (g_new, out_new)):
                resid.copy_(r0)
                side_buf.copy_(s0)
                barrier()
                graph.replay()
                torch.cuda.synchronize()
                outs.append([o.clone() for o in out])
            same &= all(torch.equal(p.view(torch.int16), q.view(torch.int16)) for p, q in zip(*outs))
        log(f"[op] as captured vs rewritten, 4 input sets: bit-identical on every rank: {all_ok(same)}")
        ok &= all_ok(same)
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = {}
        for name, graph in (("as captured", g_plain), ("rewritten", g_new)):
            ts = []
            for _ in range(25):
                barrier()
                st.record()
                graph.replay()
                en.record()
                en.synchronize()
                ts.append(st.elapsed_time(en) * 1000 / LAYERS)
            times[name] = statistics.median(ts)
        log("[op] replay per layer: " + ", ".join(f"{k} {v:.2f} us" for k, v in times.items()))
        del g_plain, g_new

        # ---- mixing: rewritten graphs replayed between eager collectives on the same stream, exact inputs
        dec_in = torch.zeros(LAYERS, T, H, dtype=torch.bfloat16)
        ag_in = torch.zeros(2, T, 6400, dtype=torch.bfloat16)
        dft_in = torch.zeros(7, T, H, dtype=torch.bfloat16)

        def dec_body():
            outs = [tensor_model_parallel_all_reduce(dec_in[i] * 1) for i in range(LAYERS)]
            gs = [tensor_model_parallel_all_gather(ag_in[j] * 1, dim=-1) for j in range(2)]
            return torch.stack(outs), torch.stack(gs)

        def dft_body():
            return (torch.stack([tensor_model_parallel_all_reduce(dft_in[i] * 1) for i in range(7)]),)

        g_dec, (dec_out, ag_out), _ = capture(dec_body, rewrite=True)
        g_dft, (dft_out,), _ = capture(dft_body, rewrite=True)

        def ints(it, salt, shape, r):
            gg = torch.Generator(device=dev).manual_seed((it * 131 + salt) * 7 + r)
            return torch.randint(-16, 17, shape, generator=gg, device=dev).to(torch.bfloat16)

        def red(it, salt, shape):
            return sum(ints(it, salt, shape, r).float() for r in range(world)).to(torch.bfloat16)

        def cat(it, salt, shape):
            return torch.cat([ints(it, salt, shape, r) for r in range(world)], dim=-1)

        bad, pending = 0, []
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for it in range(200):
                dec_in.copy_(ints(it, 1, dec_in.shape, rank))
                ag_in.copy_(ints(it, 2, ag_in.shape, rank))
                g_dec.replay()
                outs = [dec_out.clone(), ag_out.clone(),
                        tensor_model_parallel_all_reduce(ints(it, 3, (T, H), rank)),
                        tensor_model_parallel_all_gather(ints(it, 4, (T, 6400), rank), dim=-1),
                        tensor_model_parallel_all_reduce(ints(it, 5, (2048, H), rank))]
                dft_in.copy_(ints(it, 6, dft_in.shape, rank))
                g_dft.replay()
                outs.append(dft_out.clone())
                pending.append((it, outs))
                if (it + 1) % 10 == 0:
                    stream.synchronize()
                    for pit, o in pending:
                        want = [red(pit, 1, dec_in.shape), cat(pit, 2, ag_in.shape), red(pit, 3, (T, H)),
                                cat(pit, 4, (T, 6400)), red(pit, 5, (2048, H)), red(pit, 6, dft_in.shape)]
                        bad += sum(int(not torch.equal(x, y)) for x, y in zip(o, want))
                    pending.clear()
        log(f"[mixing] rewritten graphs between eager collectives, 200 iterations: wrong outputs on rank 0: {bad}")
        ok &= all_ok(bad == 0)
        del g_dec, g_dft

        # ---- integration: the patched CUDAGraphWrapper
        import vllm.compilation.cuda_graph as cg

        if not hasattr(cg, "_moet_nccl"):
            log("[integration] vllm/compilation/cuda_graph.py is not patched (patch_vllm_nccl_graph_records.py): skipped")
        else:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            from vllm.compilation.monitor import set_cudagraph_capturing_enabled
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import BatchDescriptor, set_forward_context

            class Body(torch.nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    resid.copy_(x)
                    side_buf.zero_()
                    r, s = body()
                    return torch.cat([r, s], dim=1)

            set_cudagraph_capturing_enabled(True)
            x = torch.zeros(T, H, dtype=torch.bfloat16)
            for env in ("1", "0"):
                os.environ["VLLM_MOET_NCCL_GRAPH_RECORDS"] = env
                wrapper = CUDAGraphWrapper(Body(), vllm_config, runtime_mode=CUDAGraphMode.FULL)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                good = True
                with torch.cuda.stream(stream), set_forward_context(
                        None, vllm_config, cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=BatchDescriptor(num_tokens=T)):
                    x.copy_(wide((T, H)))
                    wrapper(x)  # capture
                    for _ in range(3):
                        x.copy_(wide((T, H)))
                        barrier()
                        got = wrapper(x).clone()  # replay
                        ref = Body()(x)  # eager
                        torch.cuda.synchronize()
                        good &= torch.equal(got.view(torch.int16), ref.view(torch.int16))
                entry = next(iter(wrapper.concrete_cudagraph_entries.values()))
                try:
                    topo = Topology(entry.cudagraph.raw_cuda_graph())
                    recs = sum(1 for nm in topo.name if nm == "event-record")
                    kept = f"graph kept, {recs} record node(s)"
                    good &= env == "1" and (recs == 1 or (world == 1 and recs == 0))
                except Exception:
                    kept = "captured without keep_graph (as before)"
                    good &= env == "0"
                log(f"[integration] VLLM_MOET_NCCL_GRAPH_RECORDS={env}: replays == eager forward, {kept}: "
                    f"{'OK' if all_ok(good) else 'UNEXPECTED'}")
                ok &= all_ok(good)
                wrapper.clear_graphs()
                torch.cuda.synchronize()
            os.environ.pop("VLLM_MOET_NCCL_GRAPH_RECORDS")

            # PIECEWISE: BreakableCUDAGraphCapture, two graph segments around an eager break
            from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

            xin = torch.zeros(T, H, dtype=torch.bfloat16)

            def pieces(cap) -> None:
                resid.copy_(xin)
                side_buf.zero_()
                layers(0, LAYERS // 2)
                (cap.add_eager if cap is not None else (lambda fn: fn()))(lambda: resid.mul_(0.5))
                layers(LAYERS // 2, LAYERS)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            good = True
            with torch.cuda.stream(stream):
                xin.copy_(wide((T, H)))
                pieces(None)
                stream.synchronize()
                cap = BreakableCUDAGraphCapture()
                with cap:
                    pieces(cap)
                segs = [s.__self__ for s in cap.segments if hasattr(s, "__self__")
                        and isinstance(s.__self__, torch.cuda.CUDAGraph)]
                recs = [sum(1 for nm in Topology(gr.raw_cuda_graph()).name if nm == "event-record") for gr in segs]
                for _ in range(3):
                    xin.copy_(wide((T, H)))
                    barrier()
                    cap.replay()
                    got = (resid.clone(), side_buf.clone())
                    pieces(None)
                    torch.cuda.synchronize()
                    good &= all(torch.equal(p.view(torch.int16), q.view(torch.int16))
                                for p, q in zip(got, (resid, side_buf)))
            good &= len(segs) == 2 and all(r == (1 if world > 1 else 0) for r in recs)
            log(f"[integration] breakable capture: {len(segs)} graph segments, record nodes per segment {recs}, "
                f"replays == eager: {'OK' if all_ok(good) else 'UNEXPECTED'}")
            ok &= all_ok(good)

            # the GPU model runner's FULL graphs (CudaGraphManager.capture) take the same two calls
            import inspect

            from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

            src = inspect.getsource(CudaGraphManager.capture)
            good = "keep_graph=moet_keep" in src and "_moet_nccl.finish_capture(graph)" in src
            log(f"[integration] CudaGraphManager.capture captures with keep_graph and finish_capture: "
                f"{'OK' if good else 'NOT PATCHED'}")
            ok &= good
    log("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
